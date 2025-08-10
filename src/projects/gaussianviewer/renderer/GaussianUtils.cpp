#include <projects/gaussianviewer/renderer/GaussianUtils.hpp>

Eigen::Matrix3f build_rotation(const Rot& r) {
    // Normalize the quaternion (r.rot[0] is real part, r.rot[1..3] are i,j,k)
    float q[4] = {r.rot[0], r.rot[1], r.rot[2], r.rot[3]};
    float norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm > 0.0f) {
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
    float r_ = q[0]; // real
    float x = q[1];
    float y = q[2];
    float z = q[3];

    Eigen::Matrix3f R;
    R(0,0) = 1 - 2*(y*y + z*z);
    R(0,1) = 2*(x*y - r_*z);
    R(0,2) = 2*(x*z + r_*y);
    R(1,0) = 2*(x*y + r_*z);
    R(1,1) = 1 - 2*(x*x + z*z);
    R(1,2) = 2*(y*z - r_*x);
    R(2,0) = 2*(x*z - r_*y);
    R(2,1) = 2*(y*z + r_*x);
    R(2,2) = 1 - 2*(x*x + y*y);
    return R;
}

Rot quaternion_from_rotation_matrix(const Eigen::Matrix3f& R) {
    Rot quat;

    float trace = R(0,0) + R(1,1) + R(2,2);

    if (trace > 0) {
        float s = 0.5f / sqrtf(trace + 1.0f);
        quat.rot[0] = 0.25f / s;  // w (real part) - assuming rot[0] is w
        quat.rot[1] = (R(2,1) - R(1,2)) * s;  // x
        quat.rot[2] = (R(0,2) - R(2,0)) * s;  // y
        quat.rot[3] = (R(1,0) - R(0,1)) * s;  // z
    } else {
        if (R(0,0) > R(1,1) && R(0,0) > R(2,2)) {
            float s = 2.0f * sqrtf(1.0f + R(0,0) - R(1,1) - R(2,2));
            quat.rot[0] = (R(2,1) - R(1,2)) / s;  // w
            quat.rot[1] = 0.25f * s;              // x
            quat.rot[2] = (R(0,1) + R(1,0)) / s;  // y
            quat.rot[3] = (R(0,2) + R(2,0)) / s;  // z
        } else if (R(1,1) > R(2,2)) {
            float s = 2.0f * sqrtf(1.0f + R(1,1) - R(0,0) - R(2,2));
            quat.rot[0] = (R(0,2) - R(2,0)) / s;  // w
            quat.rot[1] = (R(0,1) + R(1,0)) / s;  // x
            quat.rot[2] = 0.25f * s;              // y
            quat.rot[3] = (R(1,2) + R(2,1)) / s;  // z
        } else {
            float s = 2.0f * sqrtf(1.0f + R(2,2) - R(0,0) - R(1,1));
            quat.rot[0] = (R(1,0) - R(0,1)) / s;  // w
            quat.rot[1] = (R(0,2) + R(2,0)) / s;  // x
            quat.rot[2] = (R(1,2) + R(2,1)) / s;  // y
            quat.rot[3] = 0.25f * s;              // z
        }
    }

    // FIX 5: Normalize the resulting quaternion
    float norm = std::sqrt(quat.rot[0]*quat.rot[0] + quat.rot[1]*quat.rot[1] +
                          quat.rot[2]*quat.rot[2] + quat.rot[3]*quat.rot[3]);
    if (norm > 1e-8f) {  // Use small epsilon instead of 0
        quat.rot[0] /= norm;
        quat.rot[1] /= norm;
        quat.rot[2] /= norm;
        quat.rot[3] /= norm;
    } else {
        // Handle degenerate case - set to identity quaternion
        quat.rot[0] = 1.0f;  // w
        quat.rot[1] = 0.0f;  // x
        quat.rot[2] = 0.0f;  // y
        quat.rot[3] = 0.0f;  // z
    }

    return quat;
}

float dot_product(const float* a, const float* b, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

std::map<int, float> compute_mean_similarities(const float* text_emb, size_t feature_dim, const NumpyArrayLoader& loader) {
    std::map<int, float> mean_similarities;
    auto array_ids = loader.getArrayIds();
    for (int array_id : array_ids) {
        const cnpy::NpyArray* array = loader.getArray(array_id);
        if (!array || array->shape.size() != 2) {
            std::cerr << "Invalid array for ID " << array_id << std::endl;
            continue;
        }
        size_t n_images = array->shape[0];
        size_t emb_dim = array->shape[1];
        if (emb_dim != feature_dim) {
            std::cerr << "Embedding dimension mismatch for ID " << array_id << ": expected " << feature_dim << ", got " << emb_dim << std::endl;
            continue;
        }
        const float* image_embeddings = loader.getFloatData(array_id);
        if (!image_embeddings) {
            std::cerr << "Failed to get data for ID " << array_id << std::endl;
            continue;
        }
        float total_similarity = 0.0f;
        for (size_t k = 0; k < n_images; ++k) {
            const float* image_emb = image_embeddings + k * emb_dim;
            float similarity = dot_product(text_emb, image_emb, emb_dim);
            total_similarity += similarity;
        }
        float mean_similarity = total_similarity / static_cast<float>(n_images);
        mean_similarities[array_id] = mean_similarity;
    }
    return mean_similarities;
}