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
#include "mesh.h"
#include "helper.h"

std::tuple<std::string, std::string, std::string> readInputFile(const std::string& filename) {
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
    std::string mesh_filename = (values.size() > 0 ? values[0] : std::string());
    std::string geometry_filename = (values.size() > 1 ? values[1] : std::string());
    std::string output_filename = (values.size() > 2 ? values[2] : std::string("out.cgns"));

    return {mesh_filename, geometry_filename, output_filename};
}

int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument

    std::cout << "Input file: " << input_filename << std::endl;

    auto [mesh_filename, geometry_filename, input_out_filename] = readInputFile(input_filename);

    std::cout << "Mesh filename: " << mesh_filename << std::endl;
    std::cout << "Geometry filename: " << geometry_filename << std::endl;
    std::cout << "Output filename (from input): " << input_out_filename << std::endl;

    // Read mesh coordinates
    Mesh mesh;
    if (!mesh.loadFromCGNS(mesh_filename)) {
        std::cerr << "Failed to load mesh: " << mesh_filename << std::endl;
        return 1;
    }
    std::cout << "--- Mesh summary ---\n" << mesh.summary << std::endl;
    std::cout << "Mesh coordinates sizes: X=" << mesh.x.size() << ", Y=" << mesh.y.size() << ", Z=" << mesh.z.size() << std::endl;
    if (mesh.x.size() > 10) std::cout << "Mesh sample X[10]=" << mesh.x[10] << std::endl;

    // printVector(mesh.cx, "Cell Centers X");
    // printVector(mesh.cy, "Cell Centers Y");
    // printVector(mesh.volume, "Cell Volumes");
    // for (auto& face : mesh.faces) {
    //     std::cout << "Face between nodes (" << face.n1 << ", " << face.n2 << "): Center=("
    //               << face.cx << ", " << face.cy << "), Normal=("
    //               << face.nx << ", " << face.ny << "), Area=" << face.area
    //               << ", leftCell=" << face.leftCell << ", rightCell=" << face.rightCell
    //               << ", isBoundary=" << face.isBoundary << std::endl;
    // }

    // Read geometry coordinates
    Mesh geom;
    if (!geom.loadFromCGNS(geometry_filename)) {
        std::cerr << "Failed to load geometry: " << geometry_filename << std::endl;
    } else {
        std::cout << "--- Geometry summary ---\n" << geom.summary << std::endl;
        std::cout << "Geometry coordinates sizes: X=" << geom.x.size() << ", Y=" << geom.y.size() << ", Z=" << geom.z.size() << std::endl;
    }

    // printVector(geom.x, "Geometry X Coordinates");
    // printVector(geom.y, "Geometry Y Coordinates");

    // Build level set on the computational mesh using geometry points
    if (geom.x.size() > 1 && mesh.ncells > 0) {
        mesh.levelSet(geom.x, geom.y);

        // quick summary: count cell types
        int nfluid=0, nsolid=0, nghost=0;
        for (int c=0;c<mesh.ncells;++c) {
            int t = mesh.cell_types[c];
            if (t == 1) ++nfluid;
            else if (t == -1) ++nsolid;
            else if (t == 0) ++nghost;
        }
        std::cout << "LevelSet summary: cells=" << mesh.ncells << ", fluid=" << nfluid << ", solid=" << nsolid << ", ghost=" << nghost << std::endl;
    }

    // identify faces by type and compute immersed-boundary normals
    mesh.assignFaceTypes();
    mesh.computeImmersedBoundaryNormals(geom.x, geom.y);

    // Assign face types based on cell types
    mesh.assignFaceTypes();

    // for (auto fid : mesh.immersedBoundaryFaces) {
    //     const auto& F = mesh.faces[fid];
    //     std::cout << fid << " "
    //               << F.cx << " " << F.cy << " "
    //               << F.ib_nx << " " << F.ib_ny << " " << F.ib_nz << "\n";
    // }
    // printVector(geom.x, "");
    // printVector(geom.y, "");

    // output filename from input file (or fallback)
    std::string out_filename = input_out_filename.empty() ? std::string("out.cgns") : input_out_filename;
    if (!mesh.writeToCGNS(out_filename)) {
        std::cerr << "Failed to write CGNS file: " << out_filename << std::endl;
    } else {
        std::cout << "Wrote CGNS: " << out_filename << std::endl;
    }


    return 0;
}