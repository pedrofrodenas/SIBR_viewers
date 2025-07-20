#include <projects/gaussianviewer/renderer/GaussianStructures.hpp>

Eigen::Matrix3f build_rotation(const Rot& r);

Rot quaternion_from_rotation_matrix(const Eigen::Matrix3f& R);