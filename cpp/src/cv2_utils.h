#pragma once

#include <opencv2/opencv.hpp>

/**
 * Expand the image into a square and fill it with the specified background color.
 *
 * @param img The input image.
 * @param background_color The color to fill the background with.
 * @return A new square image with the original image centered and the background filled.
 */
cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color);
