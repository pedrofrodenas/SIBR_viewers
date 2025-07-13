#pragma once

#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <thread>

#include <projects/gaussianviewer/renderer/GaussianStructures.hpp>



class GaussianClassifier {
public:
    GaussianClassifier(const std::string& onnx_path);

    // Process a single STFT frame
    void processGaussians(std::vector<Objects> &objectData, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& logits);

private:
    // ONNX Runtime objects
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};