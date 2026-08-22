#include "utils.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

cv::Mat load_image(
    const std::string& image_path)
{
    cv::Mat image =
        cv::imread(image_path);

    if (image.empty())
    {
        throw std::runtime_error(
            "Failed to load image: " +
            image_path
        );
    }

    return image;
}

void save_image(
    const cv::Mat& image,
    const std::string& output_path)
{
    const std::filesystem::path path(
        output_path
    );

    if (!path.parent_path().empty())
    {
        std::filesystem::create_directories(
            path.parent_path()
        );
    }

    if (!cv::imwrite(
            output_path,
            image))
    {
        throw std::runtime_error(
            "Failed to save image: " +
            output_path
        );
    }
}

std::vector<Detection> map_detections_to_original(
    const std::vector<Detection>& detections,
    const LetterboxInfo& letterbox,
    int original_width,
    int original_height)
{
    std::vector<Detection> mapped;

    mapped.reserve(
        detections.size()
    );

    for (const auto& detection : detections)
    {
        Detection result = detection;

        result.x1 =
            (detection.x1 -
             letterbox.pad_x) /
            letterbox.scale;

        result.y1 =
            (detection.y1 -
             letterbox.pad_y) /
            letterbox.scale;

        result.x2 =
            (detection.x2 -
             letterbox.pad_x) /
            letterbox.scale;

        result.y2 =
            (detection.y2 -
             letterbox.pad_y) /
            letterbox.scale;

        result.x1 = std::clamp(
            result.x1,
            0.0f,
            static_cast<float>(
                original_width - 1
            )
        );

        result.y1 = std::clamp(
            result.y1,
            0.0f,
            static_cast<float>(
                original_height - 1
            )
        );

        result.x2 = std::clamp(
            result.x2,
            0.0f,
            static_cast<float>(
                original_width - 1
            )
        );

        result.y2 = std::clamp(
            result.y2,
            0.0f,
            static_cast<float>(
                original_height - 1
            )
        );

        mapped.push_back(result);
    }

    return mapped;
}

void draw_detections(
    cv::Mat& image,
    const std::vector<Detection>& detections)
{
    for (const auto& detection : detections)
    {
        const cv::Point top_left(
            static_cast<int>(detection.x1),
            static_cast<int>(detection.y1)
        );

        const cv::Point bottom_right(
            static_cast<int>(detection.x2),
            static_cast<int>(detection.y2)
        );

        cv::rectangle(
            image,
            top_left,
            bottom_right,
            cv::Scalar(0, 255, 0),
            2
        );

        const std::string label =
            "Person " +
            std::to_string(
                detection.confidence
            );

        cv::putText(
            image,
            label,
            cv::Point(
                top_left.x,
                std::max(
                    top_left.y - 10,
                    20
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 0),
            2
        );
    }
}