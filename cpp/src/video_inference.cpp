#include "video_inference.hpp"

#include "benchmark.hpp"
#include "postprocessing.hpp"
#include "preprocessing.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>


namespace
{

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;


void process_frame(
    cv::Mat& frame,
    InferenceSession& session,
    Benchmark* benchmark
)
{
    // --------------------------------
    // Preprocessing
    // --------------------------------

    if (benchmark)
    {
        benchmark->start(
            BenchmarkStage::Preprocess
        );
    }

    LetterboxInfo letterbox;

    cv::Mat processed =
        preprocess_image(
            frame,
            letterbox
        );

    std::vector<float> input_tensor =
        image_to_tensor(processed);

    const std::vector<int64_t> input_shape = {
        1,
        3,
        INPUT_HEIGHT,
        INPUT_WIDTH
    };

    if (benchmark)
    {
        benchmark->stop(
            BenchmarkStage::Preprocess
        );
    }

    // --------------------------------
    // Inference
    // --------------------------------

    if (benchmark)
    {
        benchmark->start(
            BenchmarkStage::Inference
        );
    }

    std::vector<int64_t> output_shape;

    std::vector<float> output =
        session.run(
            input_tensor,
            input_shape,
            output_shape
        );

    if (benchmark)
    {
        benchmark->stop(
            BenchmarkStage::Inference
        );
    }

    // --------------------------------
    // Post-processing
    // --------------------------------

    if (benchmark)
    {
        benchmark->start(
            BenchmarkStage::Postprocess
        );
    }

    std::vector<Detection> detections =
        postprocess(
            output.data(),
            output_shape,
            0.25f,
            0.45f
        );

    detections =
        map_detections_to_original(
            detections,
            letterbox,
            frame.cols,
            frame.rows
        );

    draw_detections(
        frame,
        detections
    );

    if (benchmark)
    {
        benchmark->stop(
            BenchmarkStage::Postprocess
        );
    }
}


} // namespace


void run_video_inference(
    const std::string& input_path,
    const std::string& output_path,
    InferenceSession& session,
    bool benchmark_enabled
)
{
    Benchmark benchmark;

    // --------------------------------
    // Open input video
    // --------------------------------

    cv::VideoCapture capture(
        input_path,
        cv::CAP_FFMPEG
    );

    if (!capture.isOpened())
    {
        throw std::runtime_error(
            "Failed to open video using FFmpeg: "
            + input_path
        );
    }

    // --------------------------------
    // Video properties
    // --------------------------------

    const double input_fps =
        capture.get(
            cv::CAP_PROP_FPS
        );

    const int width =
        static_cast<int>(
            capture.get(
                cv::CAP_PROP_FRAME_WIDTH
            )
        );

    const int height =
        static_cast<int>(
            capture.get(
                cv::CAP_PROP_FRAME_HEIGHT
            )
        );

    const int total_frames =
        static_cast<int>(
            capture.get(
                cv::CAP_PROP_FRAME_COUNT
            )
        );

    if (width <= 0 || height <= 0)
    {
        throw std::runtime_error(
            "Invalid video dimensions"
        );
    }

    const double output_fps =
        input_fps > 0.0
            ? input_fps
            : 30.0;

    // --------------------------------
    // Create output writer
    //
    // NOTE:
    // Encoding is NOT included in the
    // benchmark timing.
    // --------------------------------

    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc(
            'm',
            'p',
            '4',
            'v'
        ),
        output_fps,
        cv::Size(
            width,
            height
        )
    );

    if (!writer.isOpened())
    {
        throw std::runtime_error(
            "Failed to create output video: "
            + output_path
        );
    }

    // --------------------------------
    // Print information
    // --------------------------------

    std::cout
        << "\n========== Video Inference ==========\n";

    std::cout
        << "Input:              "
        << input_path
        << '\n';

    std::cout
        << "Resolution:         "
        << width
        << "x"
        << height
        << '\n';

    std::cout
        << "Input FPS:          "
        << input_fps
        << '\n';

    std::cout
        << "Total frames:       "
        << total_frames
        << '\n';

    std::cout
        << "Backend:            "
        << capture.getBackendName()
        << '\n';

    std::cout
        << "Benchmark:          "
        << (
            benchmark_enabled
                ? "enabled"
                : "disabled"
        )
        << '\n';

    std::cout
        << "Output:             "
        << output_path
        << "\n\n";


    // --------------------------------
    // Benchmark
    // --------------------------------

    std::size_t processed_frames = 0;

    const auto total_start =
        std::chrono::steady_clock::now();

    cv::Mat frame;

    while (true)
    {
        // --------------------------------
        // Decode
        // --------------------------------

        if (benchmark_enabled)
        {
            benchmark.start(
                BenchmarkStage::Decode
            );
        }

        const bool frame_read =
            capture.read(frame);

        if (benchmark_enabled)
        {
            benchmark.stop(
                BenchmarkStage::Decode
            );
        }

        if (!frame_read)
        {
            break;
        }

        // --------------------------------
        // Complete frame processing
        // --------------------------------

        process_frame(
            frame,
            session,
            benchmark_enabled
                ? &benchmark
                : nullptr
        );

        // --------------------------------
        // Write output
        //
        // NOT BENCHMARKED
        // --------------------------------

        writer.write(frame);

        ++processed_frames;

        // --------------------------------
        // Progress
        // --------------------------------

        if (
            processed_frames % 10 == 0 ||
            (
                total_frames > 0 &&
                processed_frames ==
                    static_cast<std::size_t>(
                        total_frames
                    )
            )
        )
        {
            const double progress =
                total_frames > 0
                    ? 100.0 *
                        static_cast<double>(
                            processed_frames
                        ) /
                        static_cast<double>(
                            total_frames
                        )
                    : 0.0;

            std::cout
                << "\rProgress: "
                << std::fixed
                << std::setprecision(1)
                << progress
                << "% ("
                << processed_frames
                << "/"
                << total_frames
                << ")"
                << std::flush;
        }
    }

    std::cout << '\n';

    const auto total_end =
        std::chrono::steady_clock::now();

    capture.release();
    writer.release();

    // --------------------------------
    // End-to-end processing time
    // --------------------------------

    const double total_seconds =
        std::chrono::duration<double>(
            total_end - total_start
        ).count();

    const double processing_fps =
        total_seconds > 0.0
            ? static_cast<double>(
                processed_frames
            ) / total_seconds
            : 0.0;

    // --------------------------------
    // Results
    // --------------------------------

    std::cout
        << "\n========== Results ==========\n";

    std::cout
        << "Processed frames:  "
        << processed_frames
        << '\n';

    std::cout
        << "Total time:        "
        << std::fixed
        << std::setprecision(4)
        << total_seconds
        << " sec\n";

    std::cout
        << "Processing FPS:    "
        << processing_fps
        << '\n';

    std::cout
        << "Video FPS:         "
        << output_fps
        << '\n';

    std::cout
        << "Output:            "
        << output_path
        << '\n';

    // --------------------------------
    // Benchmark report
    // --------------------------------

    if (benchmark_enabled)
    {
        benchmark.set_total_frames(
            processed_frames
        );

        benchmark.print_report();

        benchmark.save_csv(
            "benchmark.csv"
        );

        std::cout
            << "\nBenchmark saved to: "
            << "benchmark.csv\n";
    }
}