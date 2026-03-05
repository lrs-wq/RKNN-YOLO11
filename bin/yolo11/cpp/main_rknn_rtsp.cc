#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string>

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
#define RTSP_URL "rtsp://192.168.40.10:554/live/0a44e7c1835643898eeedca67f5c5006_1_0"
#define MODEL_PATH "/userdata/zhongliu_project/rknn_model_zoo/examples/yolo11/model/yolo11_pitch_best.rknn" 
#define MAX_DETECTIONS 64
#define SAVE_INTERVAL 50

// 重连策略配置
#define RECONNECT_INITIAL_DELAY_US 200000   // 初始等待 0.2s (200ms)
#define RECONNECT_MAX_DELAY_US     5000000  // 最大等待 5s
#define STABLE_THRESHOLD_MS        10000    // 如果连接运行超过10秒，视为连接稳定，重置下次重连时间
// ===========================================

// 全局变量 (仅保留跨帧或跨连接需要复用的内存)
static uint8_t *nv12_compact_buffer = nullptr;
static uint8_t *rgb_compact_buffer = nullptr;
static int last_width = 0;
static int last_height = 0;
static int frame_idx_counter = 0;

static long get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
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
    // 简化打印，避免刷屏太快
    printf("[Frame %d] Time: %ld ms | Objects: %d\n", frame_idx_counter, infer_time, count);
}

// 处理单帧逻辑
static void process_frame_internal(AVFrame *frame) {
    int width = frame->width;
    int height = frame->height;

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

    if (count >= 0) {
        print_detections(results, count, end_time - start_time);
        frame_idx_counter++; 
        
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
// 返回值: 0 表示正常结束(虽然RTSP很少正常结束)，-1 表示连接错误或中断
int process_rtsp_stream() {
    AVFormatContext *fmt_ctx = nullptr;
    AVCodecContext *decoder_ctx = nullptr;
    AVPacket *pkt = nullptr;
    AVFrame *frame = nullptr;
    int video_stream_idx = -1;
    int ret_val = 0;

    // 1. 网络初始化
    AVDictionary *options = NULL;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "buffer_size", "1024000", 0);
    av_dict_set(&options, "stimeout", "5000000", 0); // 5秒超时

    fmt_ctx = avformat_alloc_context();
    
    printf("Connecting to RTSP stream: %s ...\n", RTSP_URL);
    if (avformat_open_input(&fmt_ctx, RTSP_URL, nullptr, &options) < 0) {
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
            } else {
                 printf("Error sending packet for decoding.\n");
            }
        }
        av_packet_unref(pkt);
    }

cleanup:
    // 3. 资源清理（每次连接断开后必须清理 FFmpeg 上下文）
    if (frame) av_frame_free(&frame);
    if (pkt) av_packet_free(&pkt);
    if (decoder_ctx) avcodec_free_context(&decoder_ctx);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
    
    return ret_val;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL); // 禁用日志缓存

    // 1. 初始化模型 (只初始化一次)
    std::cout << "Initializing YOLO model..." << std::endl;
    if (yolo11_init(MODEL_PATH, MAX_DETECTIONS) != 0) {
        std::cerr << "Failed to init YOLO model! Path: " << MODEL_PATH << std::endl;
        return -1;
    }
    
    avformat_network_init();

    // 2. 无限重连循环
    useconds_t current_delay_us = RECONNECT_INITIAL_DELAY_US;

    while (true) {
        long start_time = get_current_time_ms();
        
        // 尝试建立连接并处理
        // 这个函数是阻塞的，直到连接断开才会返回
        process_rtsp_stream();

        long end_time = get_current_time_ms();
        long duration = end_time - start_time;

        // 3. 重连策略判断
        printf("Connection lost.\n");

        // 如果连接持续时间超过了稳定阈值（比如10秒），说明之前的连接是好的，
        // 这次断开是偶发，重置延迟时间回初始值 0.2s
        if (duration > STABLE_THRESHOLD_MS) {
            printf("Previous connection was stable (>%dms). Resetting reconnect delay.\n", STABLE_THRESHOLD_MS);
            current_delay_us = RECONNECT_INITIAL_DELAY_US;
        }

        // 执行等待
        printf("Reconnecting in %.1f seconds...\n", (float)current_delay_us / 1000000.0f);
        usleep(current_delay_us);

        // 计算下一次等待时间 (指数退避: 0.2 -> 0.4 -> 0.8 ... -> 5.0)
        current_delay_us *= 2;
        if (current_delay_us > RECONNECT_MAX_DELAY_US) {
            current_delay_us = RECONNECT_MAX_DELAY_US;
        }
    }

    // 理论上不会运行到这里，除非手动强制 kill
    yolo11_cleanup();
    if (rgb_compact_buffer) free(rgb_compact_buffer);
    if (nv12_compact_buffer) free(nv12_compact_buffer);

    return 0;
}