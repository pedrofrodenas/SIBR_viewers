#include <projects/gaussianviewer/renderer/GaussianHull.hpp>

float calculatePercentile(const Eigen::VectorXf& column, float percentile) {
    std::vector<float> sorted_data(column.data(), column.data() + column.size());
    std::sort(sorted_data.begin(), sorted_data.end());

    float index = (percentile / 100.0f) * (sorted_data.size() - 1);
    int lower_index = static_cast<int>(std::floor(index));
    int upper_index = static_cast<int>(std::ceil(index));

    if (lower_index == upper_index) {
        return sorted_data[lower_index];
    } else {
        float weight = index - lower_index;
        return sorted_data[lower_index] * (1.0f - weight) + sorted_data[upper_index] * weight;
    }
}

Eigen::MatrixXf removeOutliers(const Eigen::MatrixXf& points, float outlier_factor) {
    int rows = points.rows();
    int cols = points.cols();

    // Calculate Q1, Q3, and IQR for each column
    Eigen::VectorXf Q1(cols);
    Eigen::VectorXf Q3(cols);
    Eigen::VectorXf IQR(cols);

    for (int col = 0; col < cols; ++col) {
        Eigen::VectorXf column = points.col(col);
        Q1(col) = calculatePercentile(column, 25.0f);
        Q3(col) = calculatePercentile(column, 75.0f);
        IQR(col) = Q3(col) - Q1(col);
    }

    // Calculate bounds for outlier detection
    Eigen::VectorXf lower_bound = Q1 - outlier_factor * IQR;
    Eigen::VectorXf upper_bound = Q3 + outlier_factor * IQR;

    Eigen::MatrixXf lower_matrix = lower_bound.transpose().replicate(rows, 1);
    Eigen::MatrixXf upper_matrix = upper_bound.transpose().replicate(rows, 1);

    Eigen::VectorX<bool> is_outlier_row = ((points.array() < lower_matrix.array()) || (points.array() > upper_matrix.array())).rowwise().any();

    // Count the number of rows to keep (inliers).
    int inlier_count = rows - static_cast<int>(is_outlier_row.count());

    if (inlier_count == 0) {
        return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(0, cols);
    }

    // Create the new matrix for the filtered points.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> filtered_points(inlier_count, cols);
    int current_row = 0;
    for (int i = 0; i < rows; ++i) {
        // If the row is NOT an outlier, copy it to the new matrix.
        if (!is_outlier_row(i)) {
            filtered_points.row(current_row++) = points.row(i);
        }
    }
    return filtered_points;
}

Eigen::Array<bool, Eigen::Dynamic, 1> points_inside_convex_hull(
    const Eigen::MatrixXf& point_cloud,
    const Eigen::MatrixXf& filtered_masked_points
) {
    using namespace Eigen;

    // Handle degenerate cases with insufficient points
    if (filtered_masked_points.rows() < 4) {
        return Array<bool, Dynamic, 1>::Constant(point_cloud.rows(), false);
    }

    // Convert to CGAL points
    std::vector<Point_3> cgal_points;
    cgal_points.reserve(filtered_masked_points.rows());
    for (int i = 0; i < filtered_masked_points.rows(); ++i) {
        cgal_points.push_back(Point_3(
            filtered_masked_points(i, 0),
            filtered_masked_points(i, 1),
            filtered_masked_points(i, 2)
        ));
    }

    // Build Delaunay triangulation
    Delaunay dt(cgal_points.begin(), cgal_points.end());

    // Create result array and check point inclusion
    Array<bool, Dynamic, 1> inside_mask(point_cloud.rows());
    for (int i = 0; i < point_cloud.rows(); ++i) {
        Point_3 p(point_cloud(i, 0), point_cloud(i, 1), point_cloud(i, 2));
        Delaunay::Locate_type lt;
        int li, lj;
        Delaunay::Cell_handle c = dt.locate(p, lt, li, lj);

        // Points are inside unless they're outside convex/affine hull
        inside_mask[i] = !(lt == Delaunay::OUTSIDE_CONVEX_HULL ||
                          lt == Delaunay::OUTSIDE_AFFINE_HULL);
    }

    return inside_mask;
}

Eigen::Array<bool, Eigen::Dynamic, 1> PointsInsideConvexHull(std::vector<Pos>& pos, Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>& mask3d)
{

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xyzCoords(
        reinterpret_cast<float*>(pos.data()), // Pointer to the first element
        pos.size(),                          // Number of rows
        3                                    // Number of columns
    );

    int num_selected = mask3d.count();

    Eigen::MatrixXf maskedPoints(num_selected, xyzCoords.cols());

    int current_row = 0;
    for (int i = 0; i < xyzCoords.rows(); ++i) {
        if (mask3d(i)) {
            maskedPoints.row(current_row++) = xyzCoords.row(i);
        }
    }

    bool remove_outliers = true;  // Set this as needed
    float outlier_factor = 1.0f;  // Equivalent to the Python outlier_factor

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> filtered_masked_points;

    if (remove_outliers) {
        filtered_masked_points = removeOutliers(maskedPoints, outlier_factor);
    } else {
        filtered_masked_points = maskedPoints;
    }

    Eigen::Array<bool, Eigen::Dynamic, 1> inside_mask =
        points_inside_convex_hull(xyzCoords, filtered_masked_points);

    return inside_mask;
}

void SpatialAwarePrunning(std::vector<Pos>& pos, Eigen::Array<bool, Eigen::Dynamic, 1>& mask3d)
{
    // Map the position data to a matrix
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xyzCoords(
        reinterpret_cast<float*>(pos.data()), pos.size(), 3);

    // Count selected points
    int num_selected = mask3d.count();

    std::cout << "Number of gaussians before spatial-aware statistical pruning: " << num_selected << std::endl;

    Eigen::MatrixXf maskedPoints(num_selected, xyzCoords.cols());

    // Fill maskedPoints with selected points and track their original indices
    std::vector<int> selected_indices;
    int current_row = 0;
    for (int i = 0; i < xyzCoords.rows(); ++i) {
        if (mask3d(i)) {
            maskedPoints.row(current_row++) = xyzCoords.row(i);
            selected_indices.push_back(i);
        }
    }

    // Calculate the centroid
    Eigen::Vector3f centroid = maskedPoints.colwise().mean();
    std::cout << "Centroid:\n" << centroid << std::endl;

    // Compute Euclidean distances to the centroid for selected points
    Eigen::VectorXf distances(num_selected);
    for (int i = 0; i < num_selected; ++i) {
        distances(i) = (maskedPoints.row(i) - centroid.transpose()).norm();
    }

    // Calculate mean and standard deviation of distances
    float mean_distance = distances.mean();
    float var_distance = ((distances.array() - mean_distance).square().sum()) / (distances.size() - 1);
    float std_distance = sqrt(var_distance);

    // Filter out points beyond 2 standard deviations
    for (int i = 0; i < num_selected; ++i) {
        if (distances(i) > mean_distance + 2 * std_distance) {
            mask3d(selected_indices[i]) = false;
        }
    }

    std::cout << "Number of gaussians after spatial-aware statistical pruning: " << mask3d.count() << std::endl;
}