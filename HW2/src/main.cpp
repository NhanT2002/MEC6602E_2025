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
#include "mesh.h"
#include "helper.h"
#include "SpatialDiscretization.h"
#include "TemporalDiscretization.h"
#include "kExactLeastSquare.h"

std::tuple<std::string, std::string, std::string, double, double, double, double, double, int> readInputFile(const std::string& filename) {
    std::ifstream input_file(filename);
    if (!input_file) {
        std::cerr << "Error opening input file: " << filename << std::endl;
        return {};
    }

    auto trim = [](std::string s) {
        // trim left
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        // trim right
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
        return s;
    };

    std::string mesh_filename;
    std::string geometry_filename;
    std::string output_filename;
    // initialize numeric values to safe defaults to avoid maybe-uninitialized warnings
    double Mach = 0.0;
    double alpha = 0.0;
    double k2 = 0.0;
    double k4 = 0.0;
    double CFL = 0.0;
    int it_max = 0;

    std::string line;
    while (std::getline(input_file, line)) {
        // remove anything after a comment character (#) and trim
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        line = trim(line);
        if (line.empty()) continue;

        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue; // skip lines without '='

        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));

        // map keys (case-sensitive as used in example)
        if (key == "mesh_filename") {
            mesh_filename = value;
        } else if (key == "geometry_filename") {
            geometry_filename = value;
        } else if (key == "output_filename") {
            output_filename = value;
        } else if (key == "Mach") {
            Mach = std::stod(value);
        } else if (key == "alpha") {
            alpha = std::stod(value);
        } else if (key == "k2") {
            k2 = std::stod(value);
        } else if (key == "k4") {
            k4 = std::stod(value);
        } else if (key == "CFL") {
            CFL = std::stod(value);
        } else if (key == "it_max") {
            it_max = std::stoi(value);
        }
    }

    if (output_filename.empty()) output_filename = "out.cgns";

    return {mesh_filename, geometry_filename, output_filename, Mach, alpha, k2, k4, CFL, it_max};
}

int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument

    std::cout << "Input file: " << input_filename << std::endl;

    auto [mesh_filename, geometry_filename, input_out_filename, Mach, alpha, k2, k4, CFL, it_max] = readInputFile(input_filename);

    std::cout << "Mesh filename: " << mesh_filename << std::endl;
    std::cout << "Geometry filename: " << geometry_filename << std::endl;
    std::cout << "Output filename (from input): " << input_out_filename << std::endl;
    std::cout << "Mach number: " << Mach << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "k2: " << k2 << std::endl;
    std::cout << "k4: " << k4 << std::endl;
    std::cout << "CFL: " << CFL << std::endl;
    std::cout << "Maximum iterations: " << it_max << std::endl;

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
    mesh.assignFaceAndCellTypes();
    mesh.computeImmersedBoundaryNormals(geom.x, geom.y);

    // for (auto fid : mesh.immersedBoundaryFaces) {
    //     const auto& F = mesh.faces[fid];
    //     std::cout << fid << " "
    //               << F.cx << " " << F.cy << " "
    //               << F.ib_nx << " " << F.ib_ny << " " << F.ib_nz << "\n";
    // }
    // printVector(geom.x, "");
    // printVector(geom.y, "");

    SpatialDiscretization FVM(mesh, Mach, alpha, k2, k4);
    std::cout << "Initialized SpatialDiscretization with Mach=" << FVM.Mach_
              << ", alpha=" << FVM.alpha_ << ", k2=" << FVM.k2_ << ", k4=" << FVM.k4_ << std::endl;
    FVM.initializeVariables();

    TemporalDiscretization solver(FVM, CFL, it_max);
    std::cout << "Initialized TemporalDiscretization with CFL=" << solver.CFL_ << " and max iterations=" << solver.it_max_ << std::endl;
    solver.solve();

    // // check if input_out_filename already exists; if so, append a number to avoid overwriting
    // std::ifstream infile_check(input_out_filename);
    // if (infile_check.good()) {
    //     infile_check.close();
    //     std::string base_name, extension;
    //     auto dot_pos = input_out_filename.find_last_of('.');
    //     if (dot_pos != std::string::npos) {
    //         base_name = input_out_filename.substr(0, dot_pos);
    //         extension = input_out_filename.substr(dot_pos);
    //     } else {
    //         base_name = input_out_filename;
    //         extension = "";
    //     }
    //     int file_index = 1;;
    //     std::string new_filename;
    //     do {
    //         new_filename = base_name + "_" + std::to_string(file_index) + extension;
    //         ++file_index;
    //     } while (std::ifstream(new_filename).good());
    //     std::cout << "Output file " << input_out_filename << " already exists. Using new filename: " << new_filename << std::endl;
    //     input_out_filename = new_filename;
    // }
    
    // if (!mesh.writeToCGNSWithCellData(input_out_filename, FVM)) {
    //     std::cerr << "Failed to write CGNS with cell data." << std::endl;
    // } else {
    //     std::cout << "Wrote CGNS with cell data: " << input_out_filename << std::endl;
    // }

    return 0;
}