#include "postprocessing.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace
{

constexpr int NUM_CLASSES = 80;
constexpr int PERSON_CLASS = 0;

float calculate_iou(const Detection& a, const Detection& b)
{
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);

    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float union_area = area_a + area_b - intersection;

    if (union_area <= 0.0f)
    {
        return 0.0f;
    }

    return intersection / union_area;
}

std::vector<Detection> apply_nms(std::vector<Detection> detections, float iou_threshold)
{
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b)
        {
            return a.confidence > b.confidence;
        }
    );

    std::vector<Detection> result;

    while (!detections.empty())
    {
        const Detection best = detections.front();
        result.push_back(best);
        detections.erase(detections.begin());

        detections.erase(std::remove_if(detections.begin(), detections.end(),
                [&](const Detection& detection){return calculate_iou(best, detection) > iou_threshold;}
            ),
            detections.end()
        );
    }

    return result;
}

} // namespace

std::vector<Detection> postprocess(const float* output, const std::vector<int64_t>& output_shape,
                                   float confidence_threshold, float nms_threshold)
{
    if (output == nullptr)
    {
        throw std::invalid_argument("Output tensor is null");
    }

    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != NUM_CLASSES + 4)
    {
        throw std::runtime_error("Unexpected YOLO output shape");
    }

    const int num_predictions = static_cast<int>(output_shape[2]);
    std::vector<Detection> detections;
    detections.reserve(num_predictions);

    for (int i = 0; i < num_predictions; ++i)
    {
        const float cx = output[i];
        const float cy = output[num_predictions + i];
        const float width = output[2 * num_predictions + i];
        const float height = output[3 * num_predictions + i];
        float best_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            const float score = output[(4 + c) * num_predictions + i];

            if (score > best_score)
            {
                best_score = score;
                best_class = c;
            }
        }

        // Only detect people.
        if (best_class != PERSON_CLASS)
        {
            continue;
        }

        if (best_score < confidence_threshold)
        {
            continue;
        }

        detections.push_back(
            Detection{
                .x1 = cx - width * 0.5f,
                .y1 = cy - height * 0.5f,
                .x2 = cx + width * 0.5f,
                .y2 = cy + height * 0.5f,
                .confidence = best_score,
                .class_id = best_class
            }
        );
    }
    return apply_nms(std::move(detections), nms_threshold);
}