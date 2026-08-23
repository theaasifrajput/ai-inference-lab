#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

enum class BenchmarkStage
{
    Decode,
    Preprocess,
    Inference,
    Postprocess,
    Encode
};

class Benchmark
{
public:
    void start(BenchmarkStage stage);

    void stop(BenchmarkStage stage);

    void record_inference(double milliseconds);

    void set_total_frames(std::size_t frames);

    void print_report() const;

    void save_csv(const std::string& path) const;

private:
    using Clock = std::chrono::steady_clock;

    struct StageStats
    {
        double total_ms = 0.0;
        std::vector<double> samples;
    };

    struct ActiveTimer
    {
        Clock::time_point start_time;
        bool running = false;
    };

    static std::string stage_name(BenchmarkStage stage);

    std::unordered_map<
        BenchmarkStage,
        StageStats
    > stats_;

    std::unordered_map<
        BenchmarkStage,
        ActiveTimer
    > active_timers_;

    std::size_t total_frames_ = 0;
};