#include "preprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;
constexpr uint8_t LETTERBOX_VALUE = 114;

} // namespace

cv::Mat preprocess_image(const cv::Mat& image, LetterboxInfo& letterbox)
{
    if (image.empty())
    {
        throw std::invalid_argument("Input image is empty");
    }

    const float scale = std::min(static_cast<float>(INPUT_WIDTH) / image.cols,
                                 static_cast<float>(INPUT_HEIGHT) / image.rows
    );

    const int resized_width = static_cast<int>(std::round(image.cols * scale));
    const int resized_height = static_cast<int>(std::round(image.rows * scale));

    letterbox.scale = scale;
    letterbox.pad_x = (INPUT_WIDTH - resized_width) / 2.0f;
    letterbox.pad_y = (INPUT_HEIGHT - resized_height) / 2.0f;

    cv::Mat resized;
    cv::resize(image, resized,cv::Size(resized_width, resized_height));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat output(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(
                   LETTERBOX_VALUE, LETTERBOX_VALUE, LETTERBOX_VALUE)
    );

    const int pad_left = static_cast<int>(std::round(letterbox.pad_x));
    const int pad_top = static_cast<int>(std::round(letterbox.pad_y));
    rgb.copyTo(output(cv::Rect(pad_left, pad_top, resized_width, resized_height)));

    return output;
}

std::vector<float> image_to_tensor(
    const cv::Mat& image)
{
    if (image.empty())
    {
        throw std::invalid_argument("Input image is empty");
    }

    const int height = image.rows;
    const int width = image.cols;
    const int channels = image.channels();

    std::vector<float> tensor(static_cast<size_t>(channels) * height * width);

    const size_t channel_size = static_cast<size_t>(height) * width;

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                const uint8_t pixel = image.at<cv::Vec3b>(h, w)[c];

                tensor[static_cast<size_t>(c) *channel_size + static_cast<size_t>(h) * width + w] =
                       static_cast<float>(pixel) / 255.0f;
            }
        }
    }

    return tensor;
}