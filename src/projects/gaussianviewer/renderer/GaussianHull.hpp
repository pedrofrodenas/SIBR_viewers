#pragma once

#include <projects/gaussianviewer/renderer/GaussianStructures.hpp>

float calculatePercentile(const Eigen::VectorXf& column, float percentile);

Eigen::MatrixXf removeOutliers(const Eigen::MatrixXf& points, float outlier_factor = 1.0f);

void PointsInsideConvexHull(std::vector<Pos>& pos, Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>& mask3d);