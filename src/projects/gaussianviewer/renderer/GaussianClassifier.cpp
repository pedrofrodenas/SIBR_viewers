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

void GaussianClassifier::processGaussians(std::vector<Objects>& objectData, std::vector<std::vector<float>>& logits)
{
    size_t count = objectData.size();

    // Prepare input data
    std::vector<float> input_data(16 * count);
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < count; ++j) {
            input_data[i * count + j] = objectData[j].objects[i];
        }
    }

    // Create input tensor
    std::vector<int64_t> input_shape = {16, static_cast<int64_t>(count), 1};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    // Define input and output names
    const char* input_names[] = {"input.1"};
    const char* output_names[] = {"8"};

    // Run inference
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Extract output tensor
    Ort::Value& output_tensor = output_tensors[0];
    const float* output_data = output_tensor.GetTensorData<float>();

    // Populate logits
    logits.resize(count);
    for (size_t j = 0; j < count; ++j) {
        logits[j].resize(6);
        for (size_t k = 0; k < 6; ++k) {
            logits[j][k] = output_data[k * count + j];
        }
    }
}