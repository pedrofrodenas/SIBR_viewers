#pragma once

#include <onnxruntime_cxx_api.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>

#include <projects/gaussianviewer/renderer/GaussianStructures.hpp>



class GaussianClassifier {
public:
    GaussianClassifier(const std::string& onnx_path);

    // Process a single STFT frame
    void processGaussians(std::vector<Objects> &objectData, std::vector<std::vector<float>>& logits);

private:
    // ONNX Runtime objects
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};