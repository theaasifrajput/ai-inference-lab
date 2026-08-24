#include "benchmark.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace
{

double percentile(std::vector<double> values, double percentile_value)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const double index = (percentile_value / 100.0) * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(index);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = index - static_cast<double>(lower);

    return values[lower] + fraction * (values[upper] - values[lower]);
}

double mean(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const double total = std::accumulate(values.begin(), values.end(), 0.0);

    return total / static_cast<double>(values.size());
}

} // namespace

std::string Benchmark::stage_name(BenchmarkStage stage)
{
    switch (stage)
    {
        case BenchmarkStage::Decode:
            return "Decode";

        case BenchmarkStage::Preprocess:
            return "Preprocess";

        case BenchmarkStage::Inference:
            return "Inference";

        case BenchmarkStage::Postprocess:
            return "Postprocess";

        case BenchmarkStage::Encode:
            return "Encode";
    }

    return "Unknown";
}

void Benchmark::start(BenchmarkStage stage)
{
    auto& timer = active_timers_[stage];
    timer.start_time = Clock::now();
    timer.running = true;
}

void Benchmark::stop(BenchmarkStage stage)
{
    auto& timer = active_timers_[stage];

    if (!timer.running)
    {
        return;
    }

    const auto end = Clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - timer.start_time).count();
    stats_[stage].samples.push_back(milliseconds);
    stats_[stage].total_ms += milliseconds;
    timer.running = false;
}

void Benchmark::record_inference(double milliseconds)
{
    stats_[BenchmarkStage::Inference].samples.push_back(milliseconds);
    stats_[BenchmarkStage::Inference].total_ms += milliseconds;
}

void Benchmark::set_total_frames(std::size_t frames)
{
    total_frames_ = frames;
}

void Benchmark::print_report() const
{
    std::cout<< "\n========== Benchmark ==========\n";
    std::cout<< "Frames: "<< total_frames_<< "\n\n";

    const BenchmarkStage stages[] =
    {
        BenchmarkStage::Decode,
        BenchmarkStage::Preprocess,
        BenchmarkStage::Inference,
        BenchmarkStage::Postprocess,
        BenchmarkStage::Encode
    };

    std::cout<< std::fixed<< std::setprecision(3);

    for (const auto stage : stages)
    {
        const auto it = stats_.find(stage);

        if (it == stats_.end() || it->second.samples.empty())
        {
            continue;
        }

        const auto& samples = it->second.samples;

        std::cout<< stage_name(stage)<< ":\n";
        std::cout<< "  Mean: "<< mean(samples)<< " ms\n";
        std::cout<< "  P50:  "<< percentile(samples, 50.0)<< " ms\n";
        std::cout<< "  P95:  "<< percentile(samples, 95.0)<< " ms\n";
        std::cout<< "  P99:  "<< percentile(samples, 99.0)<< " ms\n";
        std::cout<< "  Min:  "<< *std::min_element(samples.begin(), samples.end())<< " ms\n";
        std::cout<< "  Max:  "<< *std::max_element(samples.begin(),samples.end())<< " ms\n\n";
    }
}

void Benchmark::save_csv(const std::string& path) const
{
    std::ofstream file(path);

    if (!file)
    {
        throw std::runtime_error("Failed to open benchmark output: " + path);
    }

    file
        << "stage,mean_ms,p50_ms,p95_ms,p99_ms,min_ms,max_ms,samples\n";

    const BenchmarkStage stages[] =
    {
        BenchmarkStage::Decode,
        BenchmarkStage::Preprocess,
        BenchmarkStage::Inference,
        BenchmarkStage::Postprocess,
        BenchmarkStage::Encode
    };

    for (const auto stage : stages)
    {
        const auto it = stats_.find(stage);

        if (it == stats_.end() || it->second.samples.empty())
        {
            continue;
        }

        const auto& samples = it->second.samples;

        file
            << stage_name(stage)
            << ","
            << mean(samples)
            << ","
            << percentile(samples, 50.0)
            << ","
            << percentile(samples, 95.0)
            << ","
            << percentile(samples, 99.0)
            << ","
            << *std::min_element(samples.begin(), samples.end())
            << ","
            << *std::max_element(samples.begin(),samples.end())
            << ","
            << samples.size()
            << "\n";
    }
}