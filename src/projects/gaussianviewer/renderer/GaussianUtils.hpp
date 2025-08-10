#include <map>

#include <projects/gaussianviewer/renderer/GaussianStructures.hpp>
#include <projects/gaussianviewer/renderer/NumpyLoader.hpp>

Eigen::Matrix3f build_rotation(const Rot& r);

Rot quaternion_from_rotation_matrix(const Eigen::Matrix3f& R);

std::map<int, float> compute_mean_similarities(const float* text_emb, size_t feature_dim, const NumpyArrayLoader& loader);