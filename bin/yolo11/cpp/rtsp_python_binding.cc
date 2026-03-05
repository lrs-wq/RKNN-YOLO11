#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <algorithm>
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
#include <opencv2/opencv.hpp>
#include "image_diff.h"

// ================= 配置区域 =================
#define MAX_DETECTIONS 64
#define SAVE_INTERVAL 0
#define RECONNECT_INITIAL_DELAY_US 200000   
#define RECONNECT_MAX_DELAY_US     5000000  
#define STABLE_THRESHOLD_MS        10000    
#define DIFF_PIXEL_FILTER 25   
#define DIFF_RATIO_THRESHOLD 0.01 
#define DYNAMIC_THRESH_ENABLE true  
#define THRESH_MAX 0.05f            
#define THRESH_MIN 0.001f           
#define MAX_BALLS  16.0f            

// 定义一个简单的结构体传给 Python
struct PyDetection {
    int id;
    float prob;
    int x1, y1, x2, y2;
};

using DetectCallback = std::function<void(long frame_id, std::vector<PyDetection>)>;

// 辅助函数：计算对齐后的 stride (按 16 字节对齐)
inline int align_16(int x) {
    return (x + 15) & (~15);
}

class YoloStreamProcessor {
public:
    YoloStreamProcessor() 
        : nv12_compact_buffer(nullptr), rgb_compact_buffer(nullptr),
          last_width(0), last_height(0), frame_idx_counter(0),
          last_object_count(0), raw_frame_counter(0) {
        avformat_network_init();
    }

    ~YoloStreamProcessor() {
        cleanup_buffers();
        yolo11_cleanup();
    }

    // 暴露给 Python 的主入口
    void start_stream(std::string rtsp_url, std::string model_path, DetectCallback callback) {
        this->python_callback = callback;
        this->last_object_count = 0;

        // 初始化 YOLO 模型
        std::cout << "Initializing model: " << model_path << std::endl;
        if (yolo11_init(model_path.c_str(), MAX_DETECTIONS) != 0) {
            std::cerr << "Failed to init YOLO model!" << std::endl;
            return;
        }

        useconds_t current_delay_us = RECONNECT_INITIAL_DELAY_US;

        while (true) {
            long start_time = get_current_time_ms();
            
            // 关键：释放 GIL，允许 Python 线程并发运行
            {
                py::gil_scoped_release release;
                process_rtsp_loop(rtsp_url);
            }
            
            long end_time = get_current_time_ms();
            long duration = end_time - start_time;

            printf("Connection lost.\n");

            if (duration > STABLE_THRESHOLD_MS) {
                printf("Previous connection was stable (>%dms). Resetting reconnect delay.\n", STABLE_THRESHOLD_MS);
                current_delay_us = RECONNECT_INITIAL_DELAY_US;
            }

            printf("Reconnecting in %.1f seconds...\n", (float)current_delay_us / 1000000.0f);
            usleep(current_delay_us);

            current_delay_us *= 2;
            if (current_delay_us > RECONNECT_MAX_DELAY_US) {
                current_delay_us = RECONNECT_MAX_DELAY_US;
            }
        }
    }

private:
    // --- 成员变量 (原全局变量) ---
    uint8_t *nv12_compact_buffer;
    uint8_t *rgb_compact_buffer;
    int last_width;
    int last_height;
    int frame_idx_counter;
    uint64_t raw_frame_counter;
    int last_object_count;
    
    ImageDiff image_diff;                 // 差分计算器实例
    std::vector<cv::Rect> last_regions;   // 上一帧检测到的有效区域
    DetectCallback python_callback;

    // --- 内部辅助函数 ---
    static long get_current_time_ms() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec * 1000 + tv.tv_usec / 1000;
    }

    float calculate_dynamic_threshold(int num_objects) {
        if (!DYNAMIC_THRESH_ENABLE) return DIFF_RATIO_THRESHOLD;
        float count = std::min(std::max((float)num_objects, 0.0f), MAX_BALLS);
        float ratio = count / MAX_BALLS;
        return THRESH_MAX - ratio * (THRESH_MAX - THRESH_MIN);
    }

    void cleanup_buffers() {
        if (rgb_compact_buffer) { free(rgb_compact_buffer); rgb_compact_buffer = nullptr; }
        if (nv12_compact_buffer) { free(nv12_compact_buffer); nv12_compact_buffer = nullptr; }
    }

    void process_frame_internal(AVFrame *frame) {
        AVFrame *sw_frame = nullptr;
        bool is_hw_frame = (frame->format == AV_PIX_FMT_DRM_PRIME || frame->hw_frames_ctx != nullptr);

        // 1. 如果是硬件帧，转换到系统内存帧 (NV12)
        if (is_hw_frame) {
            sw_frame = av_frame_alloc();
            // av_hwframe_transfer_data 会将硬件 buffer 数据拷贝到系统内存
            if (av_hwframe_transfer_data(sw_frame, frame, 0) < 0) {
                std::cerr << "Error transferring the data to system memory" << std::endl;
                av_frame_free(&sw_frame);
                return; 
            }
            // 同步元数据
            sw_frame->width = frame->width;
            sw_frame->height = frame->height;
        } else {
            sw_frame = frame; // 已经是系统内存帧
        }
        int width = sw_frame->width;
        int height = sw_frame->height;

        if (!sw_frame->data[0] || width <= 0 || height <= 0) return;

        // 1. 图像差分检测
        bool should_process = true; 
        raw_frame_counter++;

        if (raw_frame_counter % 25 != 0) {
            try {
                cv::Mat current_gray(height, width, CV_8UC1, sw_frame->data[0], sw_frame->linesize[0]);
                float dynamic_ratio_thresh = calculate_dynamic_threshold(last_object_count);
                
                bool is_changed = image_diff.isChanged(current_gray, last_regions, DIFF_PIXEL_FILTER, dynamic_ratio_thresh);
                
                if (!last_regions.empty() && !is_changed) {
                    should_process = false;
                }
            } catch (const std::exception& e) {
                std::cerr << "[Warning] Diff check failed: " << e.what() << std::endl;
                should_process = true;
            }
        }

        if (!should_process) return;
        
        int aligned_width = align_16(width);
        int rgb_size = aligned_width * height * 3;
        int nv12_size = aligned_width * height * 3 / 2; // NV12 大小计算
        // 内存分配与检查
        if (rgb_compact_buffer == nullptr || width != last_width || height != last_height) {
            cleanup_buffers();
            last_width = width;
            last_height = height;
            // 这里分配稍微大一点的对齐内存，防止溢出
            // 使用 posix_memalign 或者简单的 malloc (只要大小够 aligned_width)
            rgb_compact_buffer = (uint8_t*)malloc(rgb_size + 4096); // +64 padding for safety
            nv12_compact_buffer = (uint8_t*)malloc(nv12_size + 4096); 
            if (!rgb_compact_buffer || !nv12_compact_buffer) {
                if (is_hw_frame) av_frame_free(&sw_frame);
                return;
            }
            printf("Buffer Re-allocated: %dx%d\n", width, height);
        }

        // 3. 拷贝 NV12 数据 (注意：源数据可能不是对齐的，但目标我们按对齐存)
        // Y Plane
        uint8_t* dst_y = nv12_compact_buffer;
        for (int i = 0; i < height; i++) 
            memcpy(nv12_compact_buffer + i * width, sw_frame->data[0] + i * sw_frame->linesize[0], width);
        for (int i = 0; i < height / 2; i++) 
            memcpy(nv12_compact_buffer + width * height + i * width, sw_frame->data[1] + i * sw_frame->linesize[1], width);

        // RGA 转换 (NV12 -> RGB)
          rga_buffer_t src_rga = wrapbuffer_virtualaddr((void*)nv12_compact_buffer, width, height, RK_FORMAT_YCbCr_420_SP, aligned_width, height);
        rga_buffer_t dst_rga = wrapbuffer_virtualaddr((void*)rgb_compact_buffer, width, height, RK_FORMAT_RGB_888, aligned_width, height);
        // int rga_ret = imcvtcolor(src_rga, dst_rga, src_rga.format, dst_rga.format);
        // if (rga_ret != IM_STATUS_SUCCESS) return;

        // YOLO 推理
        if (imcvtcolor(src_rga, dst_rga, src_rga.format, dst_rga.format) == IM_STATUS_SUCCESS) {
            detection_result_t* results = nullptr;
            long start_time = get_current_time_ms();
            int count = yolo11_predict(rgb_compact_buffer, width, height, 3, &results);
            long end_time = get_current_time_ms();

            // 更新状态供下一帧使用
            last_regions.clear();
            last_object_count = (count >= 0) ? count : 0;

            if (count >= 0) {
                std::vector<PyDetection> py_results;
                py_results.reserve(count);
                for (int i = 0; i < count; i++) {
                    py_results.push_back({
                        results[i].class_id, results[i].confidence,
                        results[i].x1, results[i].y1, results[i].x2, results[i].y2
                    });
                    last_regions.emplace_back(results[i].x1, results[i].y1, 
                                            results[i].x2 - results[i].x1, results[i].y2 - results[i].y1);
                }
                
                if (python_callback) {
                    py::gil_scoped_acquire acquire; 
                    python_callback(frame_idx_counter, py_results);
                }

                if(frame_idx_counter % 25 == 0) {
                    printf("[Frame %d] Time: %ld ms | Objects: %d\n", frame_idx_counter, end_time - start_time, count);
                }
                
                if (SAVE_INTERVAL > 0 && (frame_idx_counter % SAVE_INTERVAL == 0)) {
                    draw_debug_and_save(width, height, count, results);
                }

                frame_idx_counter = (frame_idx_counter + 1) % 10001;
            }
        }
        if (is_hw_frame) {
            av_frame_free(&sw_frame);
        }
    }

    void draw_debug_and_save(int width, int height, int count, detection_result_t* results) {
        int stride = width * 3;
        for (int i = 0; i < count; i++) {
            draw_rectangle(rgb_compact_buffer, width, height, stride, 
                           results[i].x1, results[i].y1, results[i].x2, results[i].y2, 255, 0, 0);
        }
        char filename[128];
        snprintf(filename, sizeof(filename), "./debug_frame_%d.ppm", frame_idx_counter);
        FILE* fp = fopen(filename, "wb");
        if (fp) {
            fprintf(fp, "P6\n%d %d\n255\n", width, height);
            for (int y = 0; y < height; y++) fwrite(rgb_compact_buffer + y * stride, 1, width * 3, fp);
            fclose(fp);
        }
    }

    void draw_rectangle(uint8_t* data, int width, int height, int linesize, 
                       int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b) {
        x1 = std::max(0, x1); y1 = std::max(0, y1);
        x2 = std::min(width - 1, x2); y2 = std::min(height - 1, y2);
        int thickness = 3;
        for (int t = 0; t < thickness; t++) {
            for (int x = x1; x <= x2; x++) {
                if (y1 + t < height) { int i = (y1 + t) * linesize + x * 3; data[i] = r; data[i+1] = g; data[i+2] = b; }
                if (y2 - t >= 0) { int i = (y2 - t) * linesize + x * 3; data[i] = r; data[i+1] = g; data[i+2] = b; }
            }
            for (int y = y1; y <= y2; y++) {
                if (x1 + t < width) { int i = y * linesize + (x1 + t) * 3; data[i] = r; data[i+1] = g; data[i+2] = b; }
                if (x2 - t >= 0) { int i = y * linesize + (x2 - t) * 3; data[i] = r; data[i+1] = g; data[i+2] = b; }
            }
        }
    }

    int process_rtsp_loop(const std::string& rtsp_url) {
        AVFormatContext *fmt_ctx = nullptr;
        AVCodecContext *decoder_ctx = nullptr;
        AVPacket *pkt = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();
        int video_stream_idx = -1;
        int ret_val = 0;

        AVDictionary *options = NULL;
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
        av_dict_set(&options, "stimeout", "5000000", 0);

        if (avformat_open_input(&fmt_ctx, rtsp_url.c_str(), nullptr, &options) < 0) {
            ret_val = -1; goto cleanup;
        }

        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
            ret_val = -1; goto cleanup;
        }

        video_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_idx < 0) {
            ret_val = -1; goto cleanup;
        }
        
        {
            AVStream *st = fmt_ctx->streams[video_stream_idx];
            const AVCodec *decoder = nullptr;
            if (st->codecpar->codec_id == AV_CODEC_ID_HEVC) decoder = avcodec_find_decoder_by_name("hevc_rkmpp");
            else if (st->codecpar->codec_id == AV_CODEC_ID_H264) decoder = avcodec_find_decoder_by_name("h264_rkmpp");
            if (!decoder) decoder = avcodec_find_decoder(st->codecpar->codec_id);
            if (!decoder) { ret_val = -1; goto cleanup; }

            decoder_ctx = avcodec_alloc_context3(decoder);
            avcodec_parameters_to_context(decoder_ctx, st->codecpar);
            av_dict_set_int(&options, "buf_mode", 1, 0);
            if (avcodec_open2(decoder_ctx, decoder, &options) < 0) { ret_val = -1; goto cleanup; }
        }

        printf("Stream connected: %s\n", rtsp_url.c_str());

        while (av_read_frame(fmt_ctx, pkt) >= 0) {
            if (pkt->stream_index == video_stream_idx) {
                if (avcodec_send_packet(decoder_ctx, pkt) >= 0) {
                    while (avcodec_receive_frame(decoder_ctx, frame) >= 0) {
                        process_frame_internal(frame);
                        av_frame_unref(frame);
                    }
                }
            }
            av_packet_unref(pkt);
        }

    cleanup:
        if (frame) av_frame_free(&frame);
        if (pkt) av_packet_free(&pkt);
        if (decoder_ctx) avcodec_free_context(&decoder_ctx);
        if (fmt_ctx) avformat_close_input(&fmt_ctx);
        return ret_val;
    }
};

// 暴露给 Python 的顶层包装函数
void start_stream_wrapper(std::string rtsp_url, std::string model_path, DetectCallback callback) {
    YoloStreamProcessor processor;
    processor.start_stream(rtsp_url, model_path, callback);
}

PYBIND11_MODULE(yolo_rtsp, m) {
    py::class_<PyDetection>(m, "Detection")
        .def_readonly("id", &PyDetection::id)
        .def_readonly("prob", &PyDetection::prob)
        .def_readonly("x1", &PyDetection::x1)
        .def_readonly("y1", &PyDetection::y1)
        .def_readonly("x2", &PyDetection::x2)
        .def_readonly("y2", &PyDetection::y2);

    m.def("start_stream", &start_stream_wrapper, 
          "Start RTSP stream and detection",
          py::arg("rtsp_url"), 
          py::arg("model_path"), 
          py::arg("callback"));
}