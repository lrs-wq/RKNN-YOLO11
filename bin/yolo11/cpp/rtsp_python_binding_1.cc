#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/imgutils.h>
}

#include <rga/im2d.h>
#include <rga/RgaApi.h>
#include "yolo_wrapper.h"

// ================= 配置区域 =================
#define MAX_DETECTIONS 64
// ===========================================
using frame_id_t = uint64_t;
struct PyDetection {
    int id;
    float prob;
    int x1, y1, x2, y2;
};
using DetectCallback = std::function<void(frame_id_t frame_id, std::vector<PyDetection>)>;

static long get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

// --- 共享资源管理器 (内存分配检查) ---
class FrameManager {
public:
    struct RgbFrame {
        std::vector<uint8_t> data;
        int width = 0;
        int height = 0;
        frame_id_t frame_id = 0;
    };

    FrameManager() {
        // 预留内存防止频繁 realloc
        buffer_ready.data.reserve(2560 * 1440 * 3);
        buffer_writing.data.reserve(2560 * 1440 * 3);
    }

    // [生产者] 获取写入缓冲
    RgbFrame* get_write_buffer(int width, int height) {
        size_t needed_size = (size_t)width * height * 3;
        
        // 不仅检查宽高，必须检查 capacity/size 是否足够
        // 因为 data 可能被消费者 swap 走变为空
        if (buffer_writing.data.size() != needed_size) {
            buffer_writing.data.resize(needed_size);
        }
        
        buffer_writing.width = width;
        buffer_writing.height = height;
        return &buffer_writing;
    }
    void reset() {
        running = true;
        has_new_frame = false;
        // 清理旧数据防止串帧
        std::lock_guard<std::mutex> lock(mtx);
        buffer_ready.frame_id = 0;
    }

    void stop() { 
        running = false; 
        cv.notify_all(); // 唤醒所有等待的线程，让它们有机会退出
    }

    // [生产者] 提交帧
    void commit_frame() {
        std::lock_guard<std::mutex> lock(mtx);
        // 交换 buffer_writing 和 buffer_ready
        // buffer_ready 变成最新的，buffer_writing 拿回旧的（可能是空的）供下次复用
        std::swap(buffer_ready, buffer_writing);
        has_new_frame = true;
        cv.notify_one(); 
    }

    // [消费者] 获取最新帧
    bool get_latest_frame(std::vector<uint8_t>& out_data, int& out_w, int& out_h, frame_id_t& out_id) {
        std::unique_lock<std::mutex> lock(mtx);
        
        if (!has_new_frame) {
            // 等待 50ms，如果没有新帧则返回 false (避免阻塞太久无法退出)
            cv.wait_for(lock, std::chrono::milliseconds(50), [this] { return has_new_frame; });
        }

        if (!has_new_frame) return false;

        // [零拷贝核心]：交换 vector 内部指针
        // 注意：buffer_ready.data 变空了，但 buffer_ready.width 还是旧值
        // 这就是为什么 get_write_buffer 必须检查 .size() 的原因
        out_data.swap(buffer_ready.data);
        
        out_w = buffer_ready.width;
        out_h = buffer_ready.height;
        out_id = buffer_ready.frame_id;
        
        has_new_frame = false;
        return true;
    }

    // void stop() { running = false; }
    bool is_running() const { return running; }

private:
    RgbFrame buffer_ready;   
    RgbFrame buffer_writing; 
    
    std::mutex mtx;
    std::condition_variable cv;
    bool has_new_frame = false;
    std::atomic<bool> running{true};
};

static FrameManager g_frame_mgr;
static std::atomic<frame_id_t> g_decode_counter{0};

// --- RGA 转换 ---
void convert_avframe_to_rgb(AVFrame *frame, uint8_t *dst_buffer) {
    if (!dst_buffer) {
        std::cerr << "Error: DST buffer is null!" << std::endl;
        return;
    }

    int width = frame->width;
    int height = frame->height;
    
    // 线程局部存储，避免每次分配临时 NV12 内存
    static thread_local std::vector<uint8_t> nv12_temp;
    size_t nv12_size = width * height * 3 / 2;
    if (nv12_temp.size() != nv12_size) {
        nv12_temp.resize(nv12_size);
    }
    
    uint8_t* compact_y = nv12_temp.data();
    uint8_t* compact_uv = compact_y + width * height;
    
    // 内存规整 (Copy Y)
    if (frame->linesize[0] == width) {
        memcpy(compact_y, frame->data[0], width * height);
    } else {
        for (int i = 0; i < height; i++) 
            memcpy(compact_y + i * width, frame->data[0] + i * frame->linesize[0], width);
    }
    // 内存规整 (Copy UV)
    if (frame->linesize[1] == width) {
        memcpy(compact_uv, frame->data[1], width * height / 2);
    } else {
        for (int i = 0; i < height / 2; i++) 
            memcpy(compact_uv + i * width, frame->data[1] + i * frame->linesize[1], width);
    }
    
    // RGA Config
    rga_buffer_t src_rga = wrapbuffer_virtualaddr((void*)compact_y, width, height, RK_FORMAT_YCbCr_420_SP, width, height);
    rga_buffer_t dst_rga = wrapbuffer_virtualaddr((void*)dst_buffer, width, height, RK_FORMAT_RGB_888, width, height);

    int ret = imcvtcolor(src_rga, dst_rga, src_rga.format, dst_rga.format);
    if (ret != IM_STATUS_SUCCESS) {
        printf("RGA Error: %d\n", ret);
    }
}

// --- 线程 1: 解码线程 ---
void decode_thread_func(std::string rtsp_url) {
    // 确保 RGA 上下文初始化 (部分环境需要)
    // RgaInit(); 

    while (g_frame_mgr.is_running()) {
        AVFormatContext *fmt_ctx = avformat_alloc_context();
        AVDictionary *options = NULL;
        // 优化：TCP 传输，极低延迟参数
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
        av_dict_set(&options, "buffer_size", "1024000", 0);
        av_dict_set(&options, "fflags", "nobuffer", 0);
        av_dict_set(&options, "flags", "low_delay", 0);
        av_dict_set(&options, "stimeout", "5000000", 0); 

        if (avformat_open_input(&fmt_ctx, rtsp_url.c_str(), nullptr, &options) < 0) {
            printf("[Decoder] Connect failed, retrying...\n");
            av_dict_free(&options);
            sleep(1);
            continue;
        }
        av_dict_free(&options);

        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
            avformat_close_input(&fmt_ctx);
            continue;
        }

        int video_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_idx < 0) {
            avformat_close_input(&fmt_ctx);
            continue;
        }

        AVStream *st = fmt_ctx->streams[video_stream_idx];
        // const AVCodec *decoder = avcodec_find_decoder_by_name("hevc_rkmpp");
        // if (!decoder) decoder = avcodec_find_decoder_by_name("h264_rkmpp");
        // if (!decoder) decoder = avcodec_find_decoder(st->codecpar->codec_id);
        const AVCodec *decoder = avcodec_find_decoder(st->codecpar->codec_id);
        AVCodecContext *decoder_ctx = avcodec_alloc_context3(decoder);
        avcodec_parameters_to_context(decoder_ctx, st->codecpar);
        
        if (avcodec_open2(decoder_ctx, decoder, NULL) < 0) {
            avcodec_free_context(&decoder_ctx);
            avformat_close_input(&fmt_ctx);
            continue;
        }

        AVPacket *pkt = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();
        
        printf("[Decoder] Stream connected. Hardware: %s\n", decoder->name);

        while (g_frame_mgr.is_running()) {
            if (av_read_frame(fmt_ctx, pkt) < 0) {
                // 读取失败或超时，检查是否需要退出
                if (!g_frame_mgr.is_running()) break;
                continue; 
            }

            if (pkt->stream_index == video_stream_idx) {
                int ret = avcodec_send_packet(decoder_ctx, pkt);
                if (ret >= 0) {
                    while (ret >= 0) {
                        //  在接收帧之前再次检查，防止在停止阶段操作硬件
                        if (!g_frame_mgr.is_running()) break;
                        ret = avcodec_receive_frame(decoder_ctx, frame);
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                        

                        // 如果已经发出了停止信号，绝对不要进行内存拷贝或 RGA 操作
                        // 这就是导致 RKMPP object -1 报错的地方
                        if (!g_frame_mgr.is_running()) {
                            av_frame_unref(frame);
                            break;
                        }
                        // 1. 获取写缓冲 (此时会自动扩容，即使是被偷走数据的 buffer)
                        auto* write_buf = g_frame_mgr.get_write_buffer(frame->width, frame->height);
                        
                        // 2. RGA 转换 YUV -> RGB (写入 write_buf)
                        convert_avframe_to_rgb(frame, write_buf->data.data());
                        write_buf->frame_id = ++g_decode_counter;
                        
                        // 3. 提交 (供推理线程读取)
                        g_frame_mgr.commit_frame();
                        
                        av_frame_unref(frame);
                    }
                }
            }
            av_packet_unref(pkt);
        }

        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&decoder_ctx);
        avformat_close_input(&fmt_ctx);
    }
}

// --- 线程 2: 推理主循环 ---
void inference_loop(std::string model_path, DetectCallback callback) {
    if (yolo11_init(model_path.c_str(), MAX_DETECTIONS) != 0) {
        std::cerr << "Failed to init YOLO model!" << std::endl;
        return;
    }

    std::vector<uint8_t> infer_buffer; 
    int width = 0, height = 0;
    frame_id_t frame_id = 0;
    detection_result_t* results = nullptr;

    while (g_frame_mgr.is_running()) {
        // 尝试获取最新帧
        if (g_frame_mgr.get_latest_frame(infer_buffer, width, height, frame_id)) {
            
            // 安全检查：如果获取的 buffer 为空 
            if (infer_buffer.empty() || width == 0 || height == 0) continue;

            long t0 = get_current_time_ms();

            // 1. 推理
            int count = yolo11_predict(infer_buffer.data(), width, height, 3, &results);
            
            long t1 = get_current_time_ms();

            // 2. Python 回调
            if (count >= 0 && callback) {
                std::vector<PyDetection> py_results;
                py_results.reserve(count);
                for (int i = 0; i < count; i++) {
                    py_results.push_back({
                        results[i].class_id,
                        results[i].confidence,
                        results[i].x1, results[i].y1, 
                        results[i].x2, results[i].y2
                    });
                }
                
                py::gil_scoped_acquire acquire;
                callback(frame_id, py_results);
            }
            long t2 = get_current_time_ms();

            // 简单的日志降频 (每 10 帧打一次)
            if (frame_id % 10 == 0) {
                 printf(">> [Frame %ld] Infer: %ld ms | PyCB: %ld ms\n", frame_id, (t1-t0), (t2-t1));
            }
        } 
    }
    yolo11_cleanup();
}

void start_stream(std::string rtsp_url, std::string model_path, DetectCallback callback) {
    // 0. 初始化状态
    g_frame_mgr.reset(); 
    avformat_network_init();

    // 1. 启动解码线程 (不要 detach!)
    std::thread decoder(decode_thread_func, rtsp_url);

    // 2. 主线程跑推理 (阻塞在这里，直到 Python 端发出停止信号或发生错误)
    // 假设 yolo11_predict 内部或者外部控制逻辑会根据某种条件退出 inference_loop
    // 或者 inference_loop 本身捕获了 KeyboardInterrupt
    try {
        py::gil_scoped_release release; 
        inference_loop(model_path, callback);
    } catch (...) {
        printf("Exception in inference loop.\n");
    }
    
    // 3. 推理循环结束了 (准备退出)
    printf("Stop signal received. Waiting for decoder to exit...\n");
    
    // 4. 通知解码线程停止
    g_frame_mgr.stop();

    // 5. 等待解码线程安全结束
    // 只有当 decode_thread_func 执行完最后的 avcodec_free_context 后，这里才会返回
    if (decoder.joinable()) {
        decoder.join();
    }
    
    printf("Stream system stopped gracefully.\n");
}

PYBIND11_MODULE(yolo_rtsp, m) {
    py::class_<PyDetection>(m, "Detection")
        .def_readonly("id", &PyDetection::id)
        .def_readonly("prob", &PyDetection::prob)
        .def_readonly("x1", &PyDetection::x1)
        .def_readonly("y1", &PyDetection::y1)
        .def_readonly("x2", &PyDetection::x2)
        .def_readonly("y2", &PyDetection::y2);

    m.def("start_stream", &start_stream, py::arg("rtsp_url"), py::arg("model_path"), py::arg("callback"));
}