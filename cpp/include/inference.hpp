#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

class InferenceSession
{
public:

    InferenceSession(
        const std::string& model_path,
        const std::string& provider
    );

    void print_model_info();

    std::vector<float> run(
        const std::vector<float>& input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>& output_shape
    );

private:

    Ort::Env env_;

    Ort::SessionOptions session_options_;

    Ort::Session session_{nullptr};
};