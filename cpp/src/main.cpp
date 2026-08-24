#include "inference.hpp"
#include "video_inference.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

struct AppConfig
{
    std::string input_path;
    std::string output_path;
    bool benchmark = false;
};

AppConfig parse_arguments(int argc, char* argv[])
{
    AppConfig config;

    if (argc == 3)
    {
        // Normal inference
        config.input_path = argv[1];
        config.output_path = argv[2];
    }
    else if (argc == 4 && std::string(argv[1]) == "--benchmark")
    {
        // Benchmark mode
        config.benchmark = true;
        config.input_path = argv[2];
        config.output_path = argv[3];
    }
    else
    {
        throw std::runtime_error(
            "Usage:\n"
            "  ai_inference <input> <output>\n"
            "  ai_inference --benchmark <input> <output>"
        );
    }

    return config;
}

} // namespace


int main(int argc, char* argv[])
{
    try
    {
        // --------------------------------------------------
        // Parse command-line arguments
        // --------------------------------------------------

        const AppConfig config = parse_arguments(argc, argv);

        // --------------------------------------------------
        // Create inference session
        // --------------------------------------------------

        InferenceSession session("models/yolov8n.onnx");

        // ----------------------------------------------------
        // Run video inference
        // -----------------------------------------------------

        run_video_inference(config.input_path, config.output_path, 
                            session, config.benchmark);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr<< "Error: "<< e.what()<< '\n';
        return 1;
    }
}