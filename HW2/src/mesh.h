#ifndef MESH_H
#define MESH_H

#include <string>
#include <tuple>
#include <vector>
#include <array>

// Simple Mesh class for structured grids (node-centered coordinates)
// and cell-centered finite-volume metric computations.
class Mesh {
public:
	// node coordinates stored row-major: index = j*ni + i
	std::vector<double> x; // size ni*nj
	std::vector<double> y;
	std::vector<double> z;
	int ni = 0; // nodes in i-direction
	int nj = 0; // nodes in j-direction

	// human-readable info
	std::string summary;

	// cell-centered quantities: number of cells = (ni-1)*(nj-1)
	int ncells = 0;
	std::vector<double> cx, cy, cz; // cell centers
	std::vector<double> volume; // cell volumes (areas in 2D)
	std::vector<int> cell_types; // cell types (1=fluid, -1=solid, 0=ghost)
	std::vector<double> phi; // signed distance function at cell centers

	struct Face {
		int n1 = -1, n2 = -1; // node indices (endpoints)
		int leftCell = -1, rightCell = -1; // adjacent cell indices (-1 for boundary)
		double cx=0, cy=0, cz=0; // face center
		double nx=0, ny=0, nz=0; // face normal (points from left->right)
		double area = 0; // face area (length in 2D)
		bool isBoundary = false;
		bool isImmersedBoundary = false;
		// immersed boundary normal (points from face center to nearest body point)
		double ib_nx = 0, ib_ny = 0, ib_nz = 0;
	};

	std::vector<Face> faces; // all unique faces
	std::vector<std::array<int,4>> cell_nodes; // node indices (4 corners) per cell
	std::vector<std::vector<int>> cell_faces; // face indices per cell (4 faces)

	std::vector<int> farfieldFaces; // indices of farfield boundary faces
	std::vector<int> immersedBoundaryFaces; // indices of immersed boundary faces
	std::vector<int> fluidFaces; // indices of fluid domain faces without boundaries

	Mesh() = default;

	// Load a structured mesh from a CGNS file (reads node coordinates and ni/nj)
	// Returns true on success.
	bool loadFromCGNS(const std::string& filename);

	bool writeToCGNS(const std::string& filename);

	// Compute cell-centered metrics (centers, faces, normals, areas) for structured grid
	// Must be called after node coordinates and ni,nj are set.
	void computeStructuredMetrics();

	void levelSet(std::vector<double> geom_x, std::vector<double> geom_y);

	// compute normals for immersed boundary faces pointing from face center to body surface
	void computeImmersedBoundaryNormals(const std::vector<double>& geom_x, const std::vector<double>& geom_y);

	void assignFaceTypes();

	// helpers
	inline int nodeIndex(int i, int j) const { return j*ni + i; }
	inline int cellIndex(int ci, int cj) const { return cj*(ni-1) + ci; }
};
#endif // MESH_H