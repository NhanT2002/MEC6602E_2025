#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include "parameters.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cgnsReader.h"

std::tuple<std::string, std::string> readInputFile(const std::string& filename) {
    std::ifstream input_file(filename);
    if (!input_file) {
        std::cerr << "Error opening input file: " << filename << std::endl;
        return {};
    }

    std::string line;
    std::vector<std::string> values;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::string key;
        char equalSign;
        if (iss >> key >> equalSign && equalSign == '=') {
            std::string value;
            if (iss >> value) {
                // std::cout << "Key: " << key << ", Value: " << value << std::endl;
                values.push_back(value);
            }
        }
    }
    std::string mesh_filename = values[0];
    std::string geometry_filename = values[1];

    return {mesh_filename, geometry_filename};
}

int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument

    std::cout << "Input file: " << input_filename << std::endl;

    auto [mesh_filename, geometry_filename] = readInputFile(input_filename);

    std::cout << "Mesh filename: " << mesh_filename << std::endl;
    std::cout << "Geometry filename: " << geometry_filename << std::endl;

    // Read mesh coordinates
    Mesh mesh = readMesh(mesh_filename);
    std::cout << "--- Mesh summary ---\n" << mesh.summary << std::endl;
    std::cout << "Mesh coordinates sizes: X=" << mesh.x.size() << ", Y=" << mesh.y.size() << ", Z=" << mesh.z.size() << std::endl;
    if (!mesh.x.empty()) std::cout << "Mesh sample X[10]=" << mesh.x[10] << std::endl;

    // Read geometry coordinates
    Mesh geom = readGeometry(geometry_filename);
    std::cout << "--- Geometry summary ---\n" << geom.summary << std::endl;
    std::cout << "Geometry coordinates sizes: X=" << geom.x.size() << ", Y=" << geom.y.size() << ", Z=" << geom.z.size() << std::endl;


    return 0;
}