#pragma once

#include "inference.hpp"

#include <string>

void run_video_inference(
    const std::string& input_path,
    const std::string& output_path,
    InferenceSession& session,
    bool benchmark_enabled = false
);