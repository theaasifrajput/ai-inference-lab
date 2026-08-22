#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

struct LetterboxInfo
{
    float scale;
    float pad_x;
    float pad_y;
};

cv::Mat preprocess_image(
    const cv::Mat& image,
    LetterboxInfo& letterbox
);

std::vector<float> image_to_tensor(
    const cv::Mat& image
);