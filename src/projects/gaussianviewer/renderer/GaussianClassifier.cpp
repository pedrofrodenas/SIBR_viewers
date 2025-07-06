#include "GaussianClassifier.hpp"
#include <iostream>

GaussianClassifier::GaussianClassifier(const std::string& onnx_path)
{
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GaussianClassifier");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ORT_SEQUENTIAL);

    int cpu_count = std::thread::hardware_concurrency();
    session_options.SetIntraOpNumThreads(cpu_count);
    session_options.SetInterOpNumThreads(std::min(4, cpu_count));

    session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();


    size_t numInputNodes = session_->GetInputCount();
    size_t numOutputNodes = session_->GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    for (size_t i = 0; i < numInputNodes; i++)
    {
        auto inputName = session_->GetInputNameAllocated(i, allocator);
        std::cout << "Input Name in " << i << " : " << inputName.get() << std::endl;
    }

    for (size_t i = 0; i < numOutputNodes; i++)
    {
        auto outputName = session_->GetOutputNameAllocated(i, allocator);
        std::cout << "Output Name in " << i << " : " << outputName.get() << std::endl;
    }
}