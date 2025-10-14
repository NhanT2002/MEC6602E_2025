#ifndef CGNSREADER_H
#define CGNSREADER_H

#include <string>
#include <tuple>
#include <vector>

struct Mesh {
	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> z;
	std::string summary; // short human-readable summary
};

// Read mesh coordinates (X,Y,Z) from the specified CGNS file. Returns a Mesh
// containing coordinate arrays (may be empty if file can't be read) and a
// summary string in Mesh::summary describing bases/zones.
Mesh readMesh(const std::string& filename);

// Read geometry coordinates (X,Y,Z) from the specified CGNS file. Same return
// type as readMesh; for many files geometry coordinates may be empty.
Mesh readGeometry(const std::string& filename);
#endif // CGNSREADER_H