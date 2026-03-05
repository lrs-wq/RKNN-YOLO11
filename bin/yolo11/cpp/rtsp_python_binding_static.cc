#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h> // 支持 python 回调

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

// 引入 Diff 头文件
#include "image_diff.h"

// ================= 配置区域 =================
#define MAX_DETECTIONS 64
#define SAVE_INTERVAL 0

// 重连策略配置
#define RECONNECT_INITIAL_DELAY_US 200000   // 初始等待 0.2s (200ms)
#define RECONNECT_MAX_DELAY_US     5000000  // 最大等待 5s
#define STABLE_THRESHOLD_MS        10000    // 稳定阈值
// ===========================================
// Diff 配置
// 像素差异阈值 (0-255)，差异大于此值才被认为是噪点外的变化
#define DIFF_PIXEL_FILTER 25   
// 变化面积比例 (0.0 - 1.0)，例如 0.01 代表区域内 1% 的像素变化才算改变
#define DIFF_RATIO_THRESHOLD 0.01 
// --- 动态阈值配置 ---
#define DYNAMIC_THRESH_ENABLE true  // 开关
#define THRESH_MAX 0.05f            // 最大值 (无目标时)
#define THRESH_MIN 0.001f           // 最小值 (满目标时)
#define MAX_BALLS  16.0f            // 达到此数量时使用最小值
// 全局变量 (仅保留跨帧或跨连接需要复用的内存)
static uint8_t *nv12_compact_buffer = nullptr;
static uint8_t *rgb_compact_buffer = nullptr;
static int last_width = 0;
static int last_height = 0;
static int frame_idx_counter = 0;
// --- Diff 相关全局变量 ---
static ImageDiff g_image_diff;                 // 差分计算器实例
static std::vector<cv::Rect> g_last_regions;   // 上一帧检测到的有效区域
// --- 记录上一帧的检测数量 ---
static int g_last_object_count = 0; 

// 定义一个简单的结构体传给 Python
struct PyDetection {
    int id;
    float prob;
    int x1, y1, x2, y2;
};

// 定义回调函数类型
using DetectCallback = std::function<void(long frame_id, std::vector<PyDetection>)>;
static DetectCallback g_python_callback = nullptr;

static long get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

// --- 动态阈值计算函数 (对应 Python 的 _calculate_dynamic_threshold) ---
static float calculate_dynamic_threshold(int num_objects) {
    if (!DYNAMIC_THRESH_ENABLE) {
        return DIFF_RATIO_THRESHOLD;
    }
    
    // 逻辑：count = min(max(num_objects, 0), MAX_BALLS)
    float count = std::min(std::max((float)num_objects, 0.0f), MAX_BALLS);
    float ratio = count / MAX_BALLS;
    
    // Formula: current_thresh = thresh_max - ratio * (thresh_max - thresh_min)
    // 对象越多(ratio越大)，阈值越小(越灵敏)
    float current_thresh = THRESH_MAX - ratio * (THRESH_MAX - THRESH_MIN);
    
    return current_thresh;
}

// --- 辅助函数：画框 ---
static void draw_rectangle(uint8_t* data, int width, int height, int linesize, 
                          int x1, int y1, int x2, int y2, 
                          uint8_t r, uint8_t g, uint8_t b) {
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 >= width) x2 = width - 1; if (y2 >= height) y2 = height - 1;

    int thickness = 3;
    for (int t = 0; t < thickness; t++) {
        for (int x = x1; x <= x2; x++) {
            if (y1 + t < height) {
                int idx = (y1 + t) * linesize + x * 3;
                data[idx] = r; data[idx+1] = g; data[idx+2] = b;
            }
            if (y2 - t >= 0) {
                int idx = (y2 - t) * linesize + x * 3;
                data[idx] = r; data[idx+1] = g; data[idx+2] = b;
            }
        }
    }
    for (int t = 0; t < thickness; t++) {
        for (int y = y1; y <= y2; y++) {
            if (x1 + t < width) {
                int idx = y * linesize + (x1 + t) * 3;
                data[idx] = r; data[idx+1] = g; data[idx+2] = b;
            }
            if (x2 - t >= 0) {
                int idx = y * linesize + (x2 - t) * 3;
                data[idx] = r; data[idx+1] = g; data[idx+2] = b;
            }
        }
    }
}

// --- 辅助函数：保存 PPM ---
static void save_debug_image(uint8_t* data, int width, int height, int linesize, int frame_id) {
    char filename[128];
    snprintf(filename, sizeof(filename), "./debug_frame_%d.ppm", frame_id);
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
        fwrite(data + y * linesize, 1, width * 3, fp);
    }
    fclose(fp);
    printf("Saved debug image: %s\n", filename);
}

static void print_detections(detection_result_t* results, int count, long infer_time) {
    printf("[Frame %d] Time: %ld ms | Objects: %d\n", frame_idx_counter, infer_time, count);
}

// 处理单帧逻辑
static void process_frame_internal(AVFrame *frame) {
    int width = frame->width;
    int height = frame->height;
    // 直接使用 AVFrame 的 Y 平面 (data[0]) 构建灰度 Mat，无需内存拷贝
    // CV_8UC1 对应灰度图
   // ---------------------------------------------------------
    // 安全性检查：如果帧数据为空或格式异常，直接跳过
    // ---------------------------------------------------------
    if (!frame->data[0] || width <= 0 || height <= 0) {
        return;
    }

    // ---------------------------------------------------------
    // 1. 图像差分检测 (增加 Try-Catch 和内存保护)
    // ---------------------------------------------------------
    bool should_process = true; // 默认处理，以防 Diff 出错

    // 静态计数器，用于控制每25帧强制检测一次
    static uint64_t raw_frame_counter = 0;
    raw_frame_counter++;

    // 逻辑：如果是第25帧，直接跳过Diff逻辑（保持should_process=true）；
    // 否则（不是第25帧），执行Diff检测。
    if (raw_frame_counter % 25 != 0) {
        try {
            // 构建 OpenCV Mat
            // 注意：frame->linesize[0] 是步长，这对 OpenCV 很重要
            cv::Mat current_gray(height, width, CV_8UC1, frame->data[0], frame->linesize[0]);
            // 使用上一帧检测到的对象数量 (g_last_object_count) 来决定当前帧的敏感度
            float dynamic_ratio_thresh = calculate_dynamic_threshold(g_last_object_count);
            // 这里的 isChanged 内部有 clone() 操作，会读取 frame->data[0]
            // 如果 data[0] 是硬件句柄，这里可能会抛出异常或导致段错误
            // 为了稳健，我们可以先尝试访问一下首字节，或者依赖 try-catch
            
            bool is_changed = g_image_diff.isChanged(current_gray, g_last_regions, DIFF_PIXEL_FILTER, dynamic_ratio_thresh);
            
            if (!g_last_regions.empty() && !is_changed) {
                should_process = false;
            }
        } catch (const cv::Exception& e) {
            // 捕获 OpenCV 内部错误，降级为"必须处理"
            std::cerr << "[Warning] Diff check failed (OpenCV Error): " << e.what() << std::endl;
            should_process = true; 
        } catch (const std::exception& e) {
            std::cerr << "[Warning] Diff check failed (Std Error): " << e.what() << std::endl;
            should_process = true;
        } catch (...) {
            std::cerr << "[Warning] Diff check failed (Unknown Error)" << std::endl;
            should_process = true;
        }
    }
    // 如果是第25帧，上面的 if 不执行，should_process 保持为 true，相当于直接返回 True 进行后续检测

    if (!should_process) {
        return; // 跳过后续昂贵的处理
    }
    // 内存分配
    if (rgb_compact_buffer == nullptr || width != last_width || height != last_height) {
        if (rgb_compact_buffer) free(rgb_compact_buffer);
        if (nv12_compact_buffer) free(nv12_compact_buffer);

        last_width = width;
        last_height = height;

        rgb_compact_buffer = (uint8_t*)malloc(width * height * 3);
        nv12_compact_buffer = (uint8_t*)malloc(width * height * 3 / 2);
        
        if (!rgb_compact_buffer || !nv12_compact_buffer) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return;
        }
        printf("Buffer Re-allocated: %dx%d\n", width, height);
    }

    // 拷贝数据到紧凑内存
    uint8_t *src_y = frame->data[0];
    uint8_t *dst_y = nv12_compact_buffer;
    for (int i = 0; i < height; i++) {
        memcpy(dst_y + i * width, src_y + i * frame->linesize[0], width);
    }
    uint8_t *src_uv = frame->data[1];
    uint8_t *dst_uv = nv12_compact_buffer + width * height; 
    for (int i = 0; i < height / 2; i++) {
        memcpy(dst_uv + i * width, src_uv + i * frame->linesize[1], width);
    }

    // RGA 转换
    rga_buffer_t src_rga = wrapbuffer_virtualaddr((void*)nv12_compact_buffer, width, height, RK_FORMAT_YCbCr_420_SP, width, height);
    rga_buffer_t dst_rga = wrapbuffer_virtualaddr((void*)rgb_compact_buffer, width, height, RK_FORMAT_RGB_888, width, height);

    if (imcvtcolor(src_rga, dst_rga, src_rga.format, dst_rga.format) != IM_STATUS_SUCCESS) {
        std::cerr << "RGA Conversion failed" << std::endl;
        return;
    }

    // YOLO 推理
    detection_result_t* results = nullptr;
    long start_time = get_current_time_ms();
    int count = yolo11_predict(rgb_compact_buffer, width, height, 3, &results);
    long end_time = get_current_time_ms();
    // =========================================================
    // 3. 更新检测区域 (为下一帧 diff 做准备)
    // =========================================================
    g_last_regions.clear(); // 清空旧区域

    // 更新全局对象计数，供下一帧 Diff 计算使用
    // 如果 count < 0 (错误)，则保持原值或重置为0，这里我们假设推理失败重置为0比较安全
    if (count >= 0) {
        g_last_object_count = count;
    } else {
        g_last_object_count = 0;
    }

    if (count >= 0) {
        if (g_python_callback) {
            std::vector<PyDetection> py_results;
            py_results.reserve(count);
            for (int i = 0; i < count; i++) {
                py_results.push_back({
                    results[i].class_id,
                    results[i].confidence,
                    results[i].x1, results[i].y1, 
                    results[i].x2, results[i].y2
                });
                // --- 填充下一次需要对比的区域 ---
                // 创建 Rect，这里直接用检测框
                g_last_regions.emplace_back(
                    results[i].x1, 
                    results[i].y1, 
                    results[i].x2 - results[i].x1, 
                    results[i].y2 - results[i].y1
                );
            }
            
            // 获取 GIL 以调用 Python 回调
            py::gil_scoped_acquire acquire; 
            g_python_callback(frame_idx_counter, py_results);
            // 作用域结束自动释放 GIL (gil_scoped_acquire 的析构)
            // 但注意：外层 process_rtsp_stream 本身是在 gil_scoped_release 块中运行的
        }
        if(frame_idx_counter % 25 == 0){
            print_detections(results, count, end_time - start_time);
        }
        frame_idx_counter++; 
        if(frame_idx_counter > 10000){
            frame_idx_counter = 0;
        }
        if (SAVE_INTERVAL > 0 && (frame_idx_counter % SAVE_INTERVAL == 0)) {
            int rgb_byte_stride = width * 3;
            for (int i = 0; i < count; i++) {
                draw_rectangle(rgb_compact_buffer, width, height, rgb_byte_stride,
                               results[i].x1, results[i].y1, results[i].x2, results[i].y2, 255, 0, 0);
            }
            save_debug_image(rgb_compact_buffer, width, height, rgb_byte_stride, frame_idx_counter);
        }
    }
}

// 封装后的单个连接处理函数
// 参数：rtsp_url - RTSP流地址
int process_rtsp_stream(const std::string& rtsp_url) {
    AVFormatContext *fmt_ctx = nullptr;
    AVCodecContext *decoder_ctx = nullptr;
    AVPacket *pkt = nullptr;
    AVFrame *frame = nullptr;
    int video_stream_idx = -1;
    int ret_val = 0;
    //连接开始前重置计数
    g_last_object_count = 0;
    // 1. 网络初始化
    AVDictionary *options = NULL;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "buffer_size", "1024000", 0);
    av_dict_set(&options, "stimeout", "5000000", 0); // 5秒超时

    fmt_ctx = avformat_alloc_context();
    
    printf("Connecting to RTSP stream: %s ...\n", rtsp_url.c_str());
    if (avformat_open_input(&fmt_ctx, rtsp_url.c_str(), nullptr, &options) < 0) {
        std::cerr << "Failed to open RTSP stream." << std::endl;
        ret_val = -1;
        goto cleanup;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "Failed to find stream info." << std::endl;
        ret_val = -1;
        goto cleanup;
    }

    video_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        std::cerr << "No video stream found." << std::endl;
        ret_val = -1;
        goto cleanup;
    }
    
    {
        AVStream *st = fmt_ctx->streams[video_stream_idx];
        const AVCodec *decoder = nullptr;
        
        if (st->codecpar->codec_id == AV_CODEC_ID_HEVC) decoder = avcodec_find_decoder_by_name("hevc_rkmpp");
        else if (st->codecpar->codec_id == AV_CODEC_ID_H264) decoder = avcodec_find_decoder_by_name("h264_rkmpp");
        if (!decoder) decoder = avcodec_find_decoder(st->codecpar->codec_id);

        if (!decoder) {
            std::cerr << "Decoder not found." << std::endl;
            ret_val = -1;
            goto cleanup;
        }

        decoder_ctx = avcodec_alloc_context3(decoder);
        avcodec_parameters_to_context(decoder_ctx, st->codecpar);
        
        AVDictionary *codec_opts = NULL;
        av_dict_set_int(&codec_opts, "buf_mode", 1, 0);
        
        if (avcodec_open2(decoder_ctx, decoder, &codec_opts) < 0) {
            std::cerr << "Failed to open codec." << std::endl;
            ret_val = -1;
            goto cleanup;
        }
    }

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    printf("Stream connected successfully. decoding...\n");

    // 2. 解码循环
    while (true) {
        int read_ret = av_read_frame(fmt_ctx, pkt);
        if (read_ret < 0) {
            printf("Stream Disconnected or EOF (Error code: %d)\n", read_ret);
            ret_val = -1;
            break; 
        }

        if (pkt->stream_index == video_stream_idx) {
            int ret = avcodec_send_packet(decoder_ctx, pkt);
            if (ret >= 0) {
                while (ret >= 0) {
                    ret = avcodec_receive_frame(decoder_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                    
                    // 处理每一帧
                    process_frame_internal(frame);
                    
                    av_frame_unref(frame);
                }
            }
        }
        av_packet_unref(pkt);
    }

cleanup:
    // 3. 资源清理
    if (frame) av_frame_free(&frame);
    if (pkt) av_packet_free(&pkt);
    if (decoder_ctx) avcodec_free_context(&decoder_ctx);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
    
    return ret_val;
}

// 暴露给 Python 的主入口
// 参数：rtsp_url, model_path, callback
void start_stream(std::string rtsp_url, std::string model_path, DetectCallback callback) {
    // 在总入口处也重置计数
    g_last_object_count = 0;
    g_python_callback = callback;
    
    // 初始化 YOLO 模型
    std::cout << "Initializing model: " << model_path << std::endl;
    // 注意：将 std::string 转为 const char* 传递给 C 接口
    if (yolo11_init(model_path.c_str(), MAX_DETECTIONS) != 0) {
        std::cerr << "Failed to init YOLO model!" << std::endl;
        return;
    }
    
    avformat_network_init();

    useconds_t current_delay_us = RECONNECT_INITIAL_DELAY_US;
    
    while (true) {
        long start_time = get_current_time_ms();
        
        // 关键：释放 GIL，允许 Python 线程并发运行，
        // 同时 C++ 在此块中阻塞进行繁重的解码和推理任务
        {
            py::gil_scoped_release release;
            // 传递 rtsp_url
            process_rtsp_stream(rtsp_url);
        }
        
        // 运行到这里说明 process_rtsp_stream 返回了（连接断开）
        long end_time = get_current_time_ms();
        long duration = end_time - start_time;

        printf("Connection lost.\n");

        if (duration > STABLE_THRESHOLD_MS) {
            printf("Previous connection was stable (>%dms). Resetting reconnect delay.\n", STABLE_THRESHOLD_MS);
            current_delay_us = RECONNECT_INITIAL_DELAY_US;
        }

        printf("Reconnecting in %.1f seconds...\n", (float)current_delay_us / 1000000.0f);
        
        // 简单起见，这里直接休眠。
        usleep(current_delay_us);

        current_delay_us *= 2;
        if (current_delay_us > RECONNECT_MAX_DELAY_US) {
            current_delay_us = RECONNECT_MAX_DELAY_US;
        }
    }

    yolo11_cleanup();
    if (rgb_compact_buffer) free(rgb_compact_buffer);
    if (nv12_compact_buffer) free(nv12_compact_buffer);
}

// 定义 Pybind11 模块
PYBIND11_MODULE(yolo_rtsp, m) {
    py::class_<PyDetection>(m, "Detection")
        .def_readonly("id", &PyDetection::id)
        .def_readonly("prob", &PyDetection::prob)
        .def_readonly("x1", &PyDetection::x1)
        .def_readonly("y1", &PyDetection::y1)
        .def_readonly("x2", &PyDetection::x2)
        .def_readonly("y2", &PyDetection::y2);

    // 修改了 start_stream 的签名，支持传入字符串参数
    m.def("start_stream", &start_stream, 
          "Start RTSP stream and detection",
          py::arg("rtsp_url"), 
          py::arg("model_path"), 
          py::arg("callback"));
}