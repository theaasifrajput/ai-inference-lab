#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <chrono>

enum class BenchmarkStage
{
    Decode,
    Preprocess,
    Inference,
    Postprocess
};


struct BenchmarkStats
{
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
};


class Benchmark
{
public:

    Benchmark() = default;

    void start(
        BenchmarkStage stage
    );

    void stop(
        BenchmarkStage stage
    );

    void set_total_frames(
        std::size_t frames
    );

    void print_report() const;

    void save_csv(
        const std::string& path
    ) const;

    std::size_t total_frames() const;

    BenchmarkStats stats(
        BenchmarkStage stage
    ) const;

    BenchmarkStats end_to_end_stats() const;

    double throughput_fps() const;

private:

    struct StageData
    {
        std::vector<double> samples;

        std::chrono::steady_clock::time_point start_time;

        bool running = false;
    };

    StageData decode_;

    StageData preprocess_;

    StageData inference_;

    StageData postprocess_;

    std::size_t total_frames_ = 0;

    static BenchmarkStats calculate_stats(
        const std::vector<double>& values
    );

    static double percentile(
        std::vector<double> values,
        double percentile
    );

    StageData& stage_data(
        BenchmarkStage stage
    );

    const StageData& stage_data(
        BenchmarkStage stage
    ) const;
};