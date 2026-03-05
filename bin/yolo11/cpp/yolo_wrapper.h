#ifndef YOLO_WRAPPER_H
#define YOLO_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 检测结果结构体
typedef struct {
    int class_id;
    float confidence;
    int x1, y1, x2, y2;
} detection_result_t;

/**
 * @brief 初始化 YOLO 模型
 * @param model_path rknn 模型文件路径
 * @param max_detections 最大检测框数量 (建议 64)
 * @return 0 成功, <0 失败
 */
int yolo11_init(const char* model_path, int max_detections);

/**
 * @brief 执行推理
 * @param image_data RGB888 图像数据指针
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数 (必须为3)
 * @param results_ptr [输出] 指向结果数组的指针
 * @return 检测到的目标数量, <0 失败
 */
int yolo11_predict(uint8_t* image_data, int width, int height, int channels, 
                   detection_result_t** results_ptr);

/**
 * @brief 清理模型资源
 */
void yolo11_cleanup();

#ifdef __cplusplus
}
#endif

#endif // YOLO_WRAPPER_H