#include "inference.hpp"
#include <iostream>

InferenceSession::InferenceSession(const std::string& model_path)
                    : env_(ORT_LOGGING_LEVEL_WARNING,"AIInference")
{
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    std::cout<< "ONNX Runtime session created\n";
}

void InferenceSession::print_model_info()
{
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_count = session_.GetInputCount();

    std::cout<< "Inputs: "<< input_count<< '\n';

    for (size_t i = 0; i < input_count;++i)
    {
        auto name = session_.GetInputNameAllocated(i, allocator);
        auto type_info = session_.GetInputTypeInfo(i);
        auto tensor_info =type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout<< "Input "<< i<< '\n';
        std::cout<< "  Name: "<< name.get()<< '\n';
        std::cout<< "  Shape: ";

        for (auto dim : shape)
        {
            std::cout<< dim<< ' ';
        }
        std::cout << '\n';
    }

    const size_t output_count = session_.GetOutputCount();

    std::cout<< "Outputs: "<< output_count<< '\n';

    for (size_t i = 0; i < output_count; ++i)
    {
        auto name = session_.GetOutputNameAllocated(i, allocator);

        auto type_info = session_.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout<< "Output "<< i<< '\n';
        std::cout<< "  Name: "<< name.get()<< '\n';
        std::cout<< "  Shape: ";

        for (auto dim : shape)
        {
            std::cout<< dim<< ' ';
        }
        std::cout << '\n';
    }
}

std::vector<float> InferenceSession::run(const std::vector<float>& input,
    const std::vector<int64_t>& input_shape, std::vector<int64_t>& output_shape)
{
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name = session_.GetInputNameAllocated(0, allocator);
    auto output_name = session_.GetOutputNameAllocated(0, allocator);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = 
            Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input.data()),
                                            input.size(),input_shape.data(),input_shape.size());

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    output_shape = output_info.GetShape();
    const float* output_data = outputs[0].GetTensorData<float>();
    const size_t output_size = output_info.GetElementCount();

    return std::vector<float>(output_data, output_data + output_size);
}