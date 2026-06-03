#include <opencv2/opencv.hpp>

cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color)
{
    const int width = img.cols;
    const int height = img.rows;

    // If the width and height are equal, return to the original image directly
    if (width == height) {
        return img.clone();
    }

    // Calculate the new size and create a new image
    const int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    // Calculate the image paste position
    const int x_offset = (size - width) / 2;
    const int y_offset = (size - height) / 2;

    // Paste the original image into the center of the new image
    const cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}
