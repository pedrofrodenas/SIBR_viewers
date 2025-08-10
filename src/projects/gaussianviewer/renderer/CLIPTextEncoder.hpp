#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

class CLIPTextEncoder {
public:
    /**
     * Constructor: Initialize the ONNX model
     * @param model_path Path to the ONNX model file
     * @param provider_name Optional provider name (default: "CPU")
     */
    explicit CLIPTextEncoder(const std::string& model_path,
                            const std::string& provider_name = "CPU");

    /**
     * Destructor
     */
    ~CLIPTextEncoder() = default;

    /**
     * Run inference on tokenized text
     * @param text_tokens Flattened token IDs (batch_size * context_length)
     * @param eot_indices End-of-text token indices for each batch item
     * @param batch_size Number of text samples in the batch
     * @param context_length Length of each tokenized sequence
     * @return Text features as float vector (batch_size * feature_dim)
     */
    std::vector<float> inference(const std::vector<int64_t>& text_tokens,
                                const std::vector<int64_t>& eot_indices,
                                size_t batch_size,
                                size_t context_length);

    /**
     * Run inference on tokenized text (alternative signature)
     * @param text_tokens 2D vector of token IDs [batch_size][context_length]
     * @param eot_indices End-of-text token indices for each batch item
     * @return Text features as float vector (batch_size * feature_dim)
     */
    std::vector<float> inference(const std::vector<std::vector<int64_t>>& text_tokens,
                                const std::vector<int64_t>& eot_indices);

    /**
     * Get the feature dimension of the output embeddings
     * @return Feature dimension size
     */
    size_t getFeatureDimension() const;

    /**
     * Check if the model is successfully loaded and ready for inference
     * @return True if model is ready, false otherwise
     */
    bool isReady() const;

    /**
     * Get model input names
     * @return Vector of input tensor names
     */
    std::vector<std::string> getInputNames() const;

    /**
     * Get model output names
     * @return Vector of output tensor names
     */
    std::vector<std::string> getOutputNames() const;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::MemoryInfo memory_info_;

    // Model metadata
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    size_t feature_dimension_;
    bool is_ready_;

    // Static input/output names for CLIP text encoder
    static const std::vector<const char*> INPUT_NAMES;
    static const std::vector<const char*> OUTPUT_NAMES;

    /**
     * Initialize model metadata (input/output names, feature dimension, etc.)
     */
    void initializeModelMetadata();

    /**
     * Validate input dimensions
     */
    bool validateInputs(const std::vector<int64_t>& text_tokens,
                       const std::vector<int64_t>& eot_indices,
                       size_t batch_size,
                       size_t context_length) const;
};