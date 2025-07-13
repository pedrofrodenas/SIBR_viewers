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

    size_t numInputNodes = session_->GetInputCount();
    size_t numOutputNodes = session_->GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    // Print input information including shapes
    for (size_t i = 0; i < numInputNodes; i++)
    {
        auto inputName = session_->GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << " Name: " << inputName.get() << std::endl;

        // Get input shape information
        auto input_info = session_->GetInputTypeInfo(i);
        auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        std::cout << "Input " << i << " Shape: [";
        for (size_t j = 0; j < input_dims.size(); j++)
        {
            std::cout << input_dims[j];
            if (j < input_dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Get data type
        auto input_type = input_tensor_info.GetElementType();
        std::cout << "Input " << i << " Type: " << input_type << std::endl;
        std::cout << "---" << std::endl;
    }

    // Print output information including shapes
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        auto outputName = session_->GetOutputNameAllocated(i, allocator);
        std::cout << "Output " << i << " Name: " << outputName.get() << std::endl;

        // Get output shape information
        auto output_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();

        std::cout << "Output " << i << " Shape: [";
        for (size_t j = 0; j < output_dims.size(); j++)
        {
            std::cout << output_dims[j];
            if (j < output_dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Get data type
        auto output_type = output_tensor_info.GetElementType();
        std::cout << "Output " << i << " Type: " << output_type << std::endl;
        std::cout << "---" << std::endl;
    }
}

void GaussianClassifier::processGaussians(std::vector<Objects>& objectData, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& logits)
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
    logits.resize(6, count);
    for (size_t k = 0; k < 6; ++k) {
        for (size_t j = 0; j < count; ++j) {
            // For tensor shape [6, count, 1], direct mapping: k * count + j
            logits(k, j) = output_data[k * count + j];
        }
    }
}