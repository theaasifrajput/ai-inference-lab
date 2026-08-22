#include "inference.hpp"
#include "postprocessing.hpp"
#include "preprocessing.hpp"
#include "utils.hpp"

#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr
            << "Usage: ai_inference <image_path>\n";

        return 1;
    }

    try
    {
        // Load image
        cv::Mat image =
            load_image(argv[1]);

        const int original_width =
            image.cols;

        const int original_height =
            image.rows;

        // Preprocessing
        LetterboxInfo letterbox;

        cv::Mat processed =
            preprocess_image(
                image,
                letterbox
            );

        std::vector<float> input_tensor =
            image_to_tensor(
                processed
            );

        const std::vector<int64_t> input_shape = {
            1,
            3,
            640,
            640
        };

        // Inference
        InferenceSession session(
            "models/yolov8n.onnx"
        );

        std::vector<int64_t> output_shape;

        std::vector<float> output =
            session.run(
                input_tensor,
                input_shape,
                output_shape
            );

        // Postprocessing
        std::vector<Detection> detections =
            postprocess(
                output.data(),
                output_shape,
                0.25f,
                0.45f
            );

        // Map model coordinates
        // back to original image
        detections =
            map_detections_to_original(
                detections,
                letterbox,
                original_width,
                original_height
            );

        // Draw
        draw_detections(
            image,
            detections
        );

        // Save
        save_image(
            image,
            "output/detected.jpg"
        );

        std::cout
            << "Detections: "
            << detections.size()
            << '\n';

        std::cout
            << "Output: "
            << "output/detected.jpg\n";
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << '\n';

        return 1;
    }

    return 0;
}