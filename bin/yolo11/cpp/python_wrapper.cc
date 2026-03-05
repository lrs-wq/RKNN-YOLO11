#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "yolo11.h"
#include "image_utils.h"

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

// 全局模型上下文
static rknn_app_context_t g_rknn_app_ctx;
static bool g_initialized = false;

// 全局图像缓冲区
static image_buffer_t g_src_image;
static bool g_image_buffer_initialized = false;
static int g_allocated_width = 0;
static int g_allocated_height = 0;
static int g_allocated_channels = 0;



// 检测结果结构体（用于C接口）
typedef struct {
    int class_id;
    float confidence;
    int x1, y1, x2, y2;
} detection_result_t;

// 全局结果缓冲区
static detection_result_t* g_results_buffer = NULL;
static int g_results_buffer_capacity = 0;
static bool g_results_buffer_initialized = false;
// 内部函数：初始化图像缓冲区
static int init_image_buffer(int width, int height, int channels) {
    if (g_image_buffer_initialized) {
        // 检查尺寸是否匹配
        if (width == g_allocated_width && height == g_allocated_height && 
            channels == g_allocated_channels) {
            return 0; // 尺寸匹配，直接使用现有缓冲区
        } else {
            printf("Image size changed! Expected: %dx%dx%d, Got: %dx%dx%d\n",
                   g_allocated_width, g_allocated_height, g_allocated_channels,
                   width, height, channels);
            return -1; // 尺寸不匹配，返回错误
        }
    }
    
    // 初始化图像缓冲区
    memset(&g_src_image, 0, sizeof(image_buffer_t));
    g_src_image.width = width;
    g_src_image.height = height;
    g_src_image.format = IMAGE_FORMAT_RGB888;
    g_src_image.size = width * height * channels;
    
    // 分配内存
    g_src_image.virt_addr = (unsigned char*)malloc(g_src_image.size);
    if (!g_src_image.virt_addr) {
        printf("Failed to allocate memory for image buffer\n");
        return -1;
    }
    
#if defined(RV1106_1103) 
    // RV1106 DMA 处理
    int ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, g_src_image.size, 
                           &g_rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                           (void **)&(g_rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
    if (ret != 0) {
        printf("Failed to allocate DMA buffer\n");
        free(g_src_image.virt_addr);
        g_src_image.virt_addr = NULL;
        return -1;
    }
    g_src_image.fd = g_rknn_app_ctx.img_dma_buf.dma_buf_fd;
    g_rknn_app_ctx.img_dma_buf.size = g_src_image.size;
#endif
    
    g_allocated_width = width;
    g_allocated_height = height;
    g_allocated_channels = channels;
    g_image_buffer_initialized = true;
    
    printf("Image buffer initialized: %dx%dx%d, size: %d\n", 
           width, height, channels, g_src_image.size);
    return 0;
}

// 内部函数：初始化结果缓冲区
static int init_results_buffer(int capacity) {
    if (g_results_buffer_initialized) {
        // 检查容量是否足够
        if (capacity <= g_results_buffer_capacity) {
            return 0; // 容量足够，直接使用现有缓冲区
        } else {
            // 需要扩容
            detection_result_t* new_buffer = (detection_result_t*)realloc(g_results_buffer, 
                                                                         capacity * sizeof(detection_result_t));
            if (!new_buffer) {
                printf("Failed to reallocate results buffer\n");
                return -1;
            }
            g_results_buffer = new_buffer;
            g_results_buffer_capacity = capacity;
            printf("Results buffer expanded to capacity: %d\n", capacity);
            return 0;
        }
    }
    
    // 首次初始化
    g_results_buffer = (detection_result_t*)malloc(capacity * sizeof(detection_result_t));
    if (!g_results_buffer) {
        printf("Failed to allocate memory for results buffer\n");
        return -1;
    }
    
    g_results_buffer_capacity = capacity;
    g_results_buffer_initialized = true;
    
    printf("Results buffer initialized with capacity: %d\n", capacity);
    return 0;
}

// 内部函数：清理图像缓冲区
static void cleanup_image_buffer() {
    if (g_image_buffer_initialized) {
        if (g_src_image.virt_addr != NULL) {
#if defined(RV1106_1103) 
            // RV1106 DMA 清理
            dma_buf_free(g_rknn_app_ctx.img_dma_buf.size, 
                        &g_rknn_app_ctx.img_dma_buf.dma_buf_fd, 
                        g_rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
            free(g_src_image.virt_addr);
#endif
            g_src_image.virt_addr = NULL;
        }
        
        g_image_buffer_initialized = false;
        g_allocated_width = 0;
        g_allocated_height = 0;
        g_allocated_channels = 0;
        
        printf("Image buffer cleaned up\n");
    }
}

// 内部函数：清理结果缓冲区
static void cleanup_results_buffer() {
    if (g_results_buffer_initialized) {
        if (g_results_buffer != NULL) {
            free(g_results_buffer);
            g_results_buffer = NULL;
        }
        
        g_results_buffer_initialized = false;
        g_results_buffer_capacity = 0;
        
        printf("Results buffer cleaned up\n");
    }
}

// 内部函数：准备图像数据
static int prepare_image_data(uint8_t* image_data, int width, int height, int channels) {
    // 初始化或检查图像缓冲区
    int ret = init_image_buffer(width, height, channels);
    if (ret != 0) {
        return ret;
    }
    
    // 复制图像数据到缓冲区
#if defined(RV1106_1103) 
    // 对于RV1106，先复制到虚拟地址，然后同步到DMA
    memcpy(g_rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, image_data, g_src_image.size);
    dma_sync_cpu_to_device(g_src_image.fd);
#else
    // 对于其他平台，直接复制
    memcpy(g_src_image.virt_addr, image_data, g_src_image.size);
#endif
    
    return 0;
}

extern "C" {

// 初始化模型
int yolo11_init(const char* model_path, int max_detections) {
    if (g_initialized) {
        printf("Model already initialized\n");
        return 0;
    }
    
    memset(&g_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    init_post_process();
    
    int ret = init_yolo11_model(model_path, &g_rknn_app_ctx);
    if (ret != 0) {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        return ret;
    }
    
    // 初始化结果缓冲区
    ret = init_results_buffer(max_detections);
    if (ret != 0) {
        printf("Failed to initialize results buffer\n");
        release_yolo11_model(&g_rknn_app_ctx);
        return ret;
    }
    
    g_initialized = true;
    return 0;
}

// 执行推理
int yolo11_predict(uint8_t* image_data, int width, int height, int channels, 
                   detection_result_t** results_ptr) {
    if (!g_initialized) {
        printf("Model not initialized\n");
        return -1;
    }
    
    if (channels != 3) {
        printf("Image must have 3 channels\n");
        return -1;
    }
    
    // 准备图像数据
    int ret = prepare_image_data(image_data, width, height, channels);
    if (ret != 0) {
        printf("Failed to prepare image data: %d\n", ret);
        return -1;
    }
    
    // 执行推理
    object_detect_result_list od_results;
    ret = inference_yolo11_model(&g_rknn_app_ctx, &g_src_image, &od_results);
    if (ret != 0) {
        printf("Inference failed: %d\n", ret);
        return -1;
    }
    
    // 检查结果数量是否超过缓冲区容量
    if (od_results.count > g_results_buffer_capacity) {
        printf("Warning: number of detections (%d) exceeds buffer capacity (%d), truncating\n",
               od_results.count, g_results_buffer_capacity);
        od_results.count = g_results_buffer_capacity;
    }
    
    // 使用预分配的结果缓冲区
    int rs_count = od_results.count < 30 ? od_results.count : 30;
    for (int i = 0; i < rs_count; i++) {
        object_detect_result *det_result = &(od_results.results[i]);
        g_results_buffer[i].class_id = det_result->cls_id;
        g_results_buffer[i].confidence = det_result->prop;
        g_results_buffer[i].x1 = det_result->box.left;
        g_results_buffer[i].y1 = det_result->box.top;
        g_results_buffer[i].x2 = det_result->box.right;
        g_results_buffer[i].y2 = det_result->box.bottom;
    }
    
    *results_ptr = g_results_buffer;
    return od_results.count;
}

// 释放检测结果内存（现在只是标记函数，实际不释放内存）
void yolo11_free_results(detection_result_t* results) {
    // 现在使用预分配的缓冲区，不需要释放
    // 这个函数保留是为了接口兼容性
    // printf("Results buffer is managed internally, no need to free\n");
}

// 清理模型资源
void yolo11_cleanup() {
    if (g_initialized) {
        release_yolo11_model(&g_rknn_app_ctx);
        deinit_post_process();
        cleanup_image_buffer();
        cleanup_results_buffer();
        g_initialized = false;
        printf("YOLO11 model cleaned up\n");
    }
}

// 可选：重新设置最大检测数量（如果需要动态调整）
int yolo11_set_max_detections(int max_detections) {
    if (!g_initialized) {
        printf("Model not initialized\n");
        return -1;
    }
    
    return init_results_buffer(max_detections);
}

} // extern "C"