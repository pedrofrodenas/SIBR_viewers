#pragma once

#include <projects/gaussianviewer/renderer/GaussianStructures.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>
#include <CGAL/Triangulation_cell_base_3.h>

template<typename Scalar>
using MatrixMap = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_3<K> Vb;
typedef CGAL::Triangulation_cell_base_3<K> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Delaunay;
typedef K::Point_3 Point_3;

float calculatePercentile(const Eigen::VectorXf& column, float percentile);

Eigen::MatrixXf removeOutliers(const Eigen::MatrixXf& points, float outlier_factor = 1.0f);

Eigen::Array<bool, Eigen::Dynamic, 1> points_inside_convex_hull(
    const Eigen::MatrixXf& point_cloud,
    const Eigen::MatrixXf& filtered_masked_points
);

Eigen::Array<bool, Eigen::Dynamic, 1> PointsInsideConvexHull(std::vector<Pos>& pos, Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>& mask3d);

void SpatialAwarePrunning(std::vector<Pos>& pos, Eigen::Array<bool, Eigen::Dynamic, 1>& mask3d);