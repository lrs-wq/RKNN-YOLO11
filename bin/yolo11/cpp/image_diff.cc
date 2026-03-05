#include "image_diff.h"

bool ImageDiff::isChanged(const cv::Mat& input, const std::vector<cv::Rect>& regions, size_t filter, double threshold) {
	bool changed = false;
	if (isChanged(input, img2, regions, filter, threshold)) {
    changed = true;
  } else if (isChanged(input, img1, regions, filter, threshold)) {
    changed = true;
  } else {
    changed = false;
  }
	// 更新帧
	img1 = img2.clone();
	img2 = input.clone();
	return changed;
}

bool ImageDiff::isChanged(const cv::Mat& imgA, const cv::Mat& imgB, const std::vector<cv::Rect>& regions, size_t filter, double threshold) {
  if (imgA.empty() || imgB.empty() || imgA.size() != imgB.size() || imgA.type() != imgB.type()) {
    return true;
  }
  for (const auto& region : regions) {
    // 限制区域在图像范围内
    cv::Rect validRegion = region & cv::Rect(0, 0, imgA.cols, imgA.rows);
    if (validRegion.width == 0 || validRegion.height == 0) {
      continue;
    }
    cv::Mat roiA = imgA(validRegion);
    cv::Mat roiB = imgB(validRegion);
    cv::Mat diff;
    cv::absdiff(roiA, roiB, diff);
    // 滤波：低于filter的像素设为0
    cv::Mat mask;
    cv::threshold(diff, mask, filter, 1, cv::THRESH_BINARY);
    int changedPixels = cv::countNonZero(mask);
    int totalPixels = validRegion.width * validRegion.height;
    double ratio = static_cast<double>(changedPixels) / totalPixels;
    if (ratio > threshold) {
      return true;
    }
  }
  return false;
}