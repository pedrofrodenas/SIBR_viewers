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

void PointsInsideConvexHull(std::vector<Pos>& pos, Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>& mask3d)
{

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> xyzCoords(
        reinterpret_cast<float*>(pos.data()), // Pointer to the first element
        pos.size(),                          // Number of rows
        3                                    // Number of columns
    );

    int num_selected = mask3d.count();
    std::cout << "Count of elements to use " << num_selected << std::endl;

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
        std::cout << "\nRemoving outliers..." << std::endl;
        filtered_masked_points = removeOutliers(maskedPoints, outlier_factor);
    } else {
        filtered_masked_points = maskedPoints;
    }

    std::cout << "\nFinal filtered points shape: " << filtered_masked_points.rows()
              << " x " << filtered_masked_points.cols() << std::endl;

    std::cout << "Hola Mundo" << std::endl;
}