#pragma once

#include <vector>
#include <map>
#include <string>
#include <cstdint>
#include <filesystem>
#include <regex>
#include <iostream>
#include <algorithm>

#include <projects/gaussianviewer/renderer/cnpy.h>

// Float16 to Float32 conversion functions
class Float16Converter {
public:
    // Convert IEEE 754 half-precision to single-precision
    static float half_to_float(uint16_t h);

    // Convert array of float16 to float32
    static std::vector<float> convert_array(const uint16_t* data, size_t count);
};

class NumpyArrayLoader {
private:
    std::map<int, cnpy::NpyArray> arrays_;
    std::map<int, std::vector<float>> converted_arrays_; // Store converted float32 data
    std::string folder_path_;

public:
    explicit NumpyArrayLoader(const std::string& folder_path);

    // Load all array_*.npy files from the specified folder
    bool loadArrays();

    // Get array by ID (returns original data)
    const cnpy::NpyArray* getArray(int id) const;

    // Get converted float32 data by ID (returns nullptr if not float16 originally)
    const std::vector<float>* getFloat32Array(int id) const;

    // Check if array was originally float16
    bool isFloat16Array(int id) const;

    // Get float32 data (either converted from float16 or original float32)
    const float* getFloatData(int id) const;

    // Get all array IDs
    std::vector<int> getArrayIds() const;

    // Get number of loaded arrays
    size_t size() const;

    // Print array information
    void printArrayInfo(int id) const;

    // Print information for all loaded arrays
    void printAllArraysInfo() const;
};
