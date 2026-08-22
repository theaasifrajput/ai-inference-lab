#pragma once

#include <cstdint>
#include <vector>

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;

    float confidence;
    int class_id;
};

std::vector<Detection> postprocess(
    const float* output,
    const std::vector<int64_t>& output_shape,
    float confidence_threshold,
    float nms_threshold
);