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

    std::string model_path =
        "models/yolov8n.onnx";

    std::string provider = "cpu";

    bool benchmark = false;
};


AppConfig parse_arguments(
    int argc,
    char* argv[]
)
{
    AppConfig config;

    if (argc == 3)
    {
        // --------------------------------------------
        // Normal inference
        //
        // ai_inference <input> <output>
        // --------------------------------------------

        config.input_path = argv[1];
        config.output_path = argv[2];
    }
    else if (
        argc == 4 &&
        std::string(argv[1]) == "--benchmark"
    )
    {
        // --------------------------------------------
        // Benchmark mode
        //
        // ai_inference
        //     --benchmark
        //     <input>
        //     <output>
        // --------------------------------------------

        config.benchmark = true;

        config.input_path = argv[2];
        config.output_path = argv[3];
    }
    else if (
        argc == 6 &&
        std::string(argv[1]) == "--benchmark" &&
        std::string(argv[4]) == "--provider"
    )
    {
        // --------------------------------------------
        // Benchmark with provider
        //
        // ai_inference
        //     --benchmark
        //     <input>
        //     <output>
        //     --provider
        //     <cpu|cuda>
        // --------------------------------------------

        config.benchmark = true;

        config.input_path = argv[2];
        config.output_path = argv[3];
        config.provider = argv[5];
    }
    else
    {
        throw std::runtime_error(
            "Usage:\n"
            "\n"
            "  ai_inference <input> <output>\n"
            "\n"
            "  ai_inference "
            "--benchmark "
            "<input> "
            "<output>\n"
            "\n"
            "  ai_inference "
            "--benchmark "
            "<input> "
            "<output> "
            "--provider "
            "<cpu|cuda>"
        );
    }

    return config;
}

} // namespace


int main(
    int argc,
    char* argv[]
)
{
    try
    {
        // --------------------------------------------------
        // Parse arguments
        // --------------------------------------------------

        const AppConfig config =
            parse_arguments(
                argc,
                argv
            );

        // --------------------------------------------------
        // Create inference session
        // --------------------------------------------------

        InferenceSession session(
            config.model_path,
            config.provider
        );

        // --------------------------------------------------
        // Run video inference
        // --------------------------------------------------

        run_video_inference(
            config.input_path,
            config.output_path,
            session,
            config.benchmark
        );

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << '\n';

        return 1;
    }
}