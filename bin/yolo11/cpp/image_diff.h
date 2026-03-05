#ifndef IMAGE_DIFF_H_
#define IMAGE_DIFF_H_

#include <opencv2/opencv.hpp>
#include <vector>

class ImageDiff {
  public:
    /*
    * 计算输入图片（灰度图）与前两帧在指定区域内是否有差异。
    * input: 输入灰度图。
    * regions: 需要对比的区域列表，上一次的有效物体检测区域。
    * filter: 滤波阈值，低于此值的差异视为无效差异。
    * threshold: 有效差异像素数量比例阈值（0-1）之间。
    */
    bool isChanged(const cv::Mat& input, const std::vector<cv::Rect>& regions, size_t filter, double threshold);
  private:
    bool isChanged(const cv::Mat& imgA, const cv::Mat& imgB, const std::vector<cv::Rect>& regions, size_t filter, double threshold);
    // 前前一帧
    cv::Mat img1;
    // 前一帧
    cv::Mat img2;
};

#endif // IMAGE_DIFF_H_