#pragma once

#include "postprocessing.hpp"
#include "preprocessing.hpp"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

cv::Mat load_image(
    const std::string& image_path
);

void save_image(
    const cv::Mat& image,
    const std::string& output_path
);

std::vector<Detection> map_detections_to_original(
    const std::vector<Detection>& detections,
    const LetterboxInfo& letterbox,
    int original_width,
    int original_height
);

void draw_detections(
    cv::Mat& image,
    const std::vector<Detection>& detections
);