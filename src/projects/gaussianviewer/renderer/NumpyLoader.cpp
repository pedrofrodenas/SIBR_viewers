#include <projects/gaussianviewer/renderer/NumpyLoader.hpp>

// Float16Converter implementations
float Float16Converter::half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h & 0x7C00) >> 10;
    uint32_t mantissa = h & 0x03FF;

    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            return *reinterpret_cast<float*>(&sign);
        } else {
            // Denormalized number
            exp = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3FF;
            exp += 127 - 15 + 1;
        }
    } else if (exp == 0x1F) {
        // Infinity or NaN
        exp = 0xFF;
    } else {
        // Normalized number
        exp += 127 - 15;
    }

    uint32_t result = sign | (exp << 23) | (mantissa << 13);
    return *reinterpret_cast<float*>(&result);
}

std::vector<float> Float16Converter::convert_array(const uint16_t* data, size_t count) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = half_to_float(data[i]);
    }
    return result;
}

// NumpyArrayLoader implementations
NumpyArrayLoader::NumpyArrayLoader(const std::string& folder_path)
    : folder_path_(folder_path) {}

bool NumpyArrayLoader::loadArrays() {
    try {
        // Check if folder exists
        if (!std::filesystem::exists(folder_path_) || !std::filesystem::is_directory(folder_path_)) {
            std::cerr << "Error: Folder " << folder_path_ << " does not exist or is not a directory" << std::endl;
            return false;
        }

        // Regular expression to match array_X.npy pattern
        std::regex pattern(R"(array_(\d+)\.npy)");
        std::smatch matches;

        // Iterate through all files in the directory
        for (const auto& entry : std::filesystem::directory_iterator(folder_path_)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();

                // Check if filename matches the pattern
                if (std::regex_match(filename, matches, pattern)) {
                    try {
                        // Extract the number from filename
                        int array_id = std::stoi(matches[1].str());

                        // Load the .npy file
                        std::string full_path = entry.path().string();
                        cnpy::NpyArray array = cnpy::npy_load(full_path);

                        // Check if this is float16 data and convert to float32
                        if (array.word_size == 2) { // float16 is 2 bytes
                            const uint16_t* data = array.data<uint16_t>();
                            converted_arrays_[array_id] = Float16Converter::convert_array(data, array.num_vals);

                            std::cout << "Loaded and converted " << filename << " (float16->float32) with ID " << array_id;
                        } else {
                            std::cout << "Loaded " << filename << " with ID " << array_id;
                        }

                        // Store original array
                        arrays_[array_id] = std::move(array);

                        std::cout << " (shape: ";
                        for (size_t i = 0; i < arrays_[array_id].shape.size(); ++i) {
                            std::cout << arrays_[array_id].shape[i];
                            if (i < arrays_[array_id].shape.size() - 1) std::cout << "x";
                        }
                        std::cout << ", word_size: " << arrays_[array_id].word_size << " bytes)" << std::endl;

                    } catch (const std::exception& e) {
                        std::cerr << "Error loading " << filename << ": " << e.what() << std::endl;
                    }
                }
            }
        }

        std::cout << "Successfully loaded " << arrays_.size() << " arrays" << std::endl;
        return !arrays_.empty();

    } catch (const std::exception& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
        return false;
    }
}

const cnpy::NpyArray* NumpyArrayLoader::getArray(int id) const {
    auto it = arrays_.find(id);
    return (it != arrays_.end()) ? &it->second : nullptr;
}

const std::vector<float>* NumpyArrayLoader::getFloat32Array(int id) const {
    auto it = converted_arrays_.find(id);
    return (it != converted_arrays_.end()) ? &it->second : nullptr;
}

bool NumpyArrayLoader::isFloat16Array(int id) const {
    const cnpy::NpyArray* array = getArray(id);
    return array && array->word_size == 2;
}

const float* NumpyArrayLoader::getFloatData(int id) const {
    if (isFloat16Array(id)) {
        const std::vector<float>* converted = getFloat32Array(id);
        return converted ? converted->data() : nullptr;
    } else {
        const cnpy::NpyArray* array = getArray(id);
        if (array && array->word_size == sizeof(float)) {
            return array->data<float>();
        }
    }
    return nullptr;
}

std::vector<int> NumpyArrayLoader::getArrayIds() const {
    std::vector<int> ids;
    for (const auto& pair : arrays_) {
        ids.push_back(pair.first);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

size_t NumpyArrayLoader::size() const {
    return arrays_.size();
}

void NumpyArrayLoader::printArrayInfo(int id) const {
    const cnpy::NpyArray* array = getArray(id);
    if (!array) {
        std::cout << "Array " << id << " not found" << std::endl;
        return;
    }

    std::cout << "Array " << id << ":" << std::endl;
    std::cout << "  Shape: ";
    for (size_t i = 0; i < array->shape.size(); ++i) {
        std::cout << array->shape[i];
        if (i < array->shape.size() - 1) std::cout << " x ";
    }
    std::cout << std::endl;
    std::cout << "  Original data type size: " << array->word_size << " bytes";
    if (array->word_size == 2) {
        std::cout << " (float16, converted to float32)";
    }
    std::cout << std::endl;
    std::cout << "  Total elements: " << array->num_vals << std::endl;

    // Print first few elements using float32 data
    const float* data = getFloatData(id);
    if (data) {
        std::cout << "  First 10 elements (float32): ";
        size_t print_count = std::min(static_cast<size_t>(10), array->num_vals);
        for (size_t i = 0; i < print_count; ++i) {
            std::cout << data[i];
            if (i < print_count - 1) std::cout << ", ";
        }
        if (array->num_vals > 10) std::cout << " ...";
        std::cout << std::endl;
    }
}

void NumpyArrayLoader::printAllArraysInfo() const {
    std::vector<int> ids = getArrayIds();
    for (int id : ids) {
        printArrayInfo(id);
        std::cout << std::endl;
    }
}