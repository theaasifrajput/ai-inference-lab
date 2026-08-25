#include "benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>


namespace
{

double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

} // namespace


// ============================================================
// Stage access
// ============================================================

Benchmark::StageData&
Benchmark::stage_data(
    BenchmarkStage stage
)
{
    switch (stage)
    {
        case BenchmarkStage::Decode:
            return decode_;

        case BenchmarkStage::Preprocess:
            return preprocess_;

        case BenchmarkStage::Inference:
            return inference_;

        case BenchmarkStage::Postprocess:
            return postprocess_;
    }

    throw std::runtime_error(
        "Invalid benchmark stage"
    );
}


const Benchmark::StageData&
Benchmark::stage_data(
    BenchmarkStage stage
) const
{
    switch (stage)
    {
        case BenchmarkStage::Decode:
            return decode_;

        case BenchmarkStage::Preprocess:
            return preprocess_;

        case BenchmarkStage::Inference:
            return inference_;

        case BenchmarkStage::Postprocess:
            return postprocess_;
    }

    throw std::runtime_error(
        "Invalid benchmark stage"
    );
}


// ============================================================
// Timing
// ============================================================

void Benchmark::start(
    BenchmarkStage stage
)
{
    StageData& data =
        stage_data(stage);

    if (data.running)
    {
        throw std::runtime_error(
            "Benchmark stage already running"
        );
    }

    data.start_time =
        std::chrono::steady_clock::now();

    data.running = true;
}


void Benchmark::stop(
    BenchmarkStage stage
)
{
    StageData& data =
        stage_data(stage);

    if (!data.running)
    {
        throw std::runtime_error(
            "Benchmark stage was not started"
        );
    }

    const auto end =
        std::chrono::steady_clock::now();

    const double milliseconds =
        elapsed_ms(
            data.start_time,
            end
        );

    data.samples.push_back(
        milliseconds
    );

    data.running = false;
}


// ============================================================
// Configuration
// ============================================================

void Benchmark::set_total_frames(
    std::size_t frames
)
{
    total_frames_ = frames;
}


std::size_t Benchmark::total_frames() const
{
    return total_frames_;
}


// ============================================================
// Statistics
// ============================================================

double Benchmark::percentile(
    std::vector<double> values,
    double p
)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    const double index =
        (values.size() - 1) * p;

    const std::size_t lower =
        static_cast<std::size_t>(
            index
        );

    const std::size_t upper =
        std::min(
            lower + 1,
            values.size() - 1
        );

    const double weight =
        index -
        static_cast<double>(
            lower
        );

    return values[lower]
        + (
            values[upper]
            - values[lower]
        ) * weight;
}


BenchmarkStats Benchmark::calculate_stats(
    const std::vector<double>& values
)
{
    BenchmarkStats result;

    if (values.empty())
    {
        return result;
    }

    result.mean =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        ) / values.size();

    result.p50 =
        percentile(
            values,
            0.50
        );

    result.p95 =
        percentile(
            values,
            0.95
        );

    result.p99 =
        percentile(
            values,
            0.99
        );

    result.min =
        *std::min_element(
            values.begin(),
            values.end()
        );

    result.max =
        *std::max_element(
            values.begin(),
            values.end()
        );

    return result;
}


BenchmarkStats Benchmark::stats(
    BenchmarkStage stage
) const
{
    return calculate_stats(
        stage_data(stage).samples
    );
}


// ============================================================
// End-to-end
// ============================================================

BenchmarkStats Benchmark::end_to_end_stats() const
{
    const std::size_t frame_count =
        std::min(
            {
                decode_.samples.size(),
                preprocess_.samples.size(),
                inference_.samples.size(),
                postprocess_.samples.size()
            }
        );

    if (frame_count == 0)
    {
        return {};
    }

    std::vector<double> total_times;

    total_times.reserve(
        frame_count
    );

    for (
        std::size_t i = 0;
        i < frame_count;
        ++i
    )
    {
        total_times.push_back(
            decode_.samples[i]
            + preprocess_.samples[i]
            + inference_.samples[i]
            + postprocess_.samples[i]
        );
    }

    return calculate_stats(
        total_times
    );
}


// ============================================================
// Throughput
// ============================================================

double Benchmark::throughput_fps() const
{
    const BenchmarkStats end_to_end =
        end_to_end_stats();

    if (end_to_end.mean <= 0.0)
    {
        return 0.0;
    }

    return 1000.0 /
        end_to_end.mean;
}


// ============================================================
// Console report
// ============================================================

void Benchmark::print_report() const
{
    std::cout
        << "\n========== Benchmark ==========\n";

    auto print_stage =
        [this](
            const std::string& name,
            BenchmarkStage stage
        )
        {
            const BenchmarkStats result =
                stats(stage);

            std::cout
                << "\n"
                << name
                << "\n";

            std::cout
                << "  Mean: "
                << std::fixed
                << std::setprecision(3)
                << result.mean
                << " ms\n";

            std::cout
                << "  P50:  "
                << result.p50
                << " ms\n";

            std::cout
                << "  P95:  "
                << result.p95
                << " ms\n";

            std::cout
                << "  P99:  "
                << result.p99
                << " ms\n";

            std::cout
                << "  Min:  "
                << result.min
                << " ms\n";

            std::cout
                << "  Max:  "
                << result.max
                << " ms\n";
        };

    print_stage(
        "Decode",
        BenchmarkStage::Decode
    );

    print_stage(
        "Preprocessing",
        BenchmarkStage::Preprocess
    );

    print_stage(
        "Inference",
        BenchmarkStage::Inference
    );

    print_stage(
        "Post-processing",
        BenchmarkStage::Postprocess
    );

    const BenchmarkStats end_to_end =
        end_to_end_stats();

    std::cout
        << "\nEnd-to-end\n";

    std::cout
        << "  Mean: "
        << end_to_end.mean
        << " ms\n";

    std::cout
        << "  P50:  "
        << end_to_end.p50
        << " ms\n";

    std::cout
        << "  P95:  "
        << end_to_end.p95
        << " ms\n";

    std::cout
        << "  P99:  "
        << end_to_end.p99
        << " ms\n";

    std::cout
        << "  Min:  "
        << end_to_end.min
        << " ms\n";

    std::cout
        << "  Max:  "
        << end_to_end.max
        << " ms\n";

    std::cout
        << "\nThroughput\n";

    std::cout
        << "  FPS:  "
        << throughput_fps()
        << '\n';

    std::cout
        << "\nFrames\n";

    std::cout
        << "  Total: "
        << total_frames_
        << '\n';
}


// ============================================================
// CSV
// ============================================================

void Benchmark::save_csv(
    const std::string& path
) const
{
    std::ofstream file(path);

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Failed to open benchmark CSV: "
            + path
        );
    }

    file
        << "frame,"
        << "decode_ms,"
        << "preprocess_ms,"
        << "inference_ms,"
        << "postprocess_ms,"
        << "end_to_end_ms\n";

    const std::size_t frame_count =
        std::min(
            {
                decode_.samples.size(),
                preprocess_.samples.size(),
                inference_.samples.size(),
                postprocess_.samples.size()
            }
        );

    for (
        std::size_t i = 0;
        i < frame_count;
        ++i
    )
    {
        const double end_to_end =
            decode_.samples[i]
            + preprocess_.samples[i]
            + inference_.samples[i]
            + postprocess_.samples[i];

        file
            << i
            << ','
            << decode_.samples[i]
            << ','
            << preprocess_.samples[i]
            << ','
            << inference_.samples[i]
            << ','
            << postprocess_.samples[i]
            << ','
            << end_to_end
            << '\n';
    }
}