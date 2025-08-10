#include "CLIPTextEncoder.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>

// Static member definitions
const std::vector<const char*> CLIPTextEncoder::INPUT_NAMES = {"text_tokens", "eot_indices"};
const std::vector<const char*> CLIPTextEncoder::OUTPUT_NAMES = {"text_features"};

CLIPTextEncoder::CLIPTextEncoder(const std::string& model_path,
                                const std::string& provider_name)
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      feature_dimension_(0),
      is_ready_(false) {

    try {
        // Initialize ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CLIPTextEncoder");

        // Create session options
        session_options_ = std::make_unique<Ort::SessionOptions>();

        // Add provider if specified (e.g., CUDA, CPU, etc.)
        if (provider_name == "CUDA") {
            // Optionally add CUDA provider configuration here
            // session_options_->AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
        }

        // Create inference session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

        // Initialize model metadata
        initializeModelMetadata();

        is_ready_ = true;
        std::cout << "CLIP Text Encoder model loaded successfully from: " << model_path << std::endl;
        std::cout << "Feature dimension: " << feature_dimension_ << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error during model loading: " << e.what()
                  << " (Code: " << e.GetOrtErrorCode() << ")" << std::endl;
        is_ready_ = false;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error loading CLIP Text Encoder model: " << e.what() << std::endl;
        is_ready_ = false;
        throw;
    }
}

std::vector<float> CLIPTextEncoder::inference(const std::vector<int64_t>& text_tokens,
                                            const std::vector<int64_t>& eot_indices,
                                            size_t batch_size,
                                            size_t context_length) {

    if (!is_ready_) {
        throw std::runtime_error("Model is not ready for inference");
    }

    if (!validateInputs(text_tokens, eot_indices, batch_size, context_length)) {
        throw std::invalid_argument("Invalid input dimensions");
    }

    try {
        // Create input tensors
        std::vector<int64_t> text_tokens_shape = {static_cast<int64_t>(batch_size),
                                                  static_cast<int64_t>(context_length)};

        Ort::Value text_tokens_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            const_cast<int64_t*>(text_tokens.data()),
            text_tokens.size(),
            text_tokens_shape.data(),
            text_tokens_shape.size()
        );

        std::vector<int64_t> eot_indices_shape = {static_cast<int64_t>(batch_size)};
        Ort::Value eot_indices_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            const_cast<int64_t*>(eot_indices.data()),
            eot_indices.size(),
            eot_indices_shape.data(),
            eot_indices_shape.size()
        );

        // Prepare inputs
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(text_tokens_tensor));
        inputs.push_back(std::move(eot_indices_tensor));

        // Run inference
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                   INPUT_NAMES.data(),
                                   inputs.data(),
                                   inputs.size(),
                                   OUTPUT_NAMES.data(),
                                   OUTPUT_NAMES.size());

        // Extract output
        Ort::Value& output_tensor = outputs[0];
        float* output_data = output_tensor.GetTensorMutableData<float>();

        // Get output shape to verify dimensions
        auto output_shape_info = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = output_shape_info.GetShape();

        if (output_shape.size() != 2 ||
            output_shape[0] != static_cast<int64_t>(batch_size) ||
            output_shape[1] != static_cast<int64_t>(feature_dimension_)) {
            throw std::runtime_error("Unexpected output tensor shape");
        }

        // Copy output to vector
        size_t total_elements = batch_size * feature_dimension_;
        std::vector<float> result(output_data, output_data + total_elements);

        return result;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error during inference: " << e.what()
                  << " (Code: " << e.GetOrtErrorCode() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> CLIPTextEncoder::inference(const std::vector<std::vector<int64_t>>& text_tokens,
                                            const std::vector<int64_t>& eot_indices) {

    if (text_tokens.empty()) {
        throw std::invalid_argument("Empty text tokens");
    }

    size_t batch_size = text_tokens.size();
    size_t context_length = text_tokens[0].size();

    // Validate all sequences have the same length
    for (const auto& seq : text_tokens) {
        if (seq.size() != context_length) {
            throw std::invalid_argument("All text sequences must have the same length");
        }
    }

    // Flatten the 2D token array
    std::vector<int64_t> flat_tokens;
    flat_tokens.reserve(batch_size * context_length);

    for (const auto& seq : text_tokens) {
        flat_tokens.insert(flat_tokens.end(), seq.begin(), seq.end());
    }

    return inference(flat_tokens, eot_indices, batch_size, context_length);
}

size_t CLIPTextEncoder::getFeatureDimension() const {
    return feature_dimension_;
}

bool CLIPTextEncoder::isReady() const {
    return is_ready_;
}

std::vector<std::string> CLIPTextEncoder::getInputNames() const {
    return input_names_;
}

std::vector<std::string> CLIPTextEncoder::getOutputNames() const {
    return output_names_;
}

void CLIPTextEncoder::initializeModelMetadata() {
    try {
        // Get input names
        size_t num_inputs = session_->GetInputCount();
        input_names_.reserve(num_inputs);

        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.emplace_back(input_name.get());
        }

        // Get output names
        size_t num_outputs = session_->GetOutputCount();
        output_names_.reserve(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.emplace_back(output_name.get());
        }

        // Get feature dimension from output shape
        if (num_outputs > 0) {
            auto output_info = session_->GetOutputTypeInfo(0);
            auto tensor_info = output_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_shape = tensor_info.GetShape();

            if (output_shape.size() >= 2 && output_shape[1] > 0) {
                feature_dimension_ = static_cast<size_t>(output_shape[1]);
            } else {
                throw std::runtime_error("Cannot determine feature dimension from model output");
            }
        }

        // Print model info
        std::cout << "Model inputs: ";
        for (const auto& name : input_names_) {
            std::cout << name << " ";
        }
        std::cout << std::endl;

        std::cout << "Model outputs: ";
        for (const auto& name : output_names_) {
            std::cout << name << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing model metadata: " << e.what() << std::endl;
        throw;
    }
}

bool CLIPTextEncoder::validateInputs(const std::vector<int64_t>& text_tokens,
                                    const std::vector<int64_t>& eot_indices,
                                    size_t batch_size,
                                    size_t context_length) const {

    // Check if dimensions match
    if (text_tokens.size() != batch_size * context_length) {
        std::cerr << "Text tokens size mismatch. Expected: " << batch_size * context_length
                  << ", got: " << text_tokens.size() << std::endl;
        return false;
    }

    if (eot_indices.size() != batch_size) {
        std::cerr << "EOT indices size mismatch. Expected: " << batch_size
                  << ", got: " << eot_indices.size() << std::endl;
        return false;
    }

    // Validate EOT indices are within valid range
    for (size_t i = 0; i < eot_indices.size(); ++i) {
        if (eot_indices[i] < 0 || eot_indices[i] >= static_cast<int64_t>(context_length)) {
            std::cerr << "Invalid EOT index at position " << i << ": " << eot_indices[i]
                      << " (should be in range [0, " << context_length - 1 << "])" << std::endl;
            return false;
        }
    }

    return true;
}