#ifndef MESH_H
#define MESH_H

#include <string>
#include <tuple>
#include <vector>
#include <array>
#include <limits>
#include "kExactLeastSquare.h"
// Forward declaration to avoid circular include: mesh.h <-> SpatialDiscretization.h
class SpatialDiscretization;

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
	std::vector<double> avg_face_area_x; // average face area in x-direction per cell
	std::vector<double> avg_face_area_y; // average face area in y-direction per cell
	std::vector<double> avg_face_area_z; // average face area in z-direction per cell
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
		double x_mirror = 0, y_mirror = 0, z_mirror = 0; // mirror point coordinates for IB faces
		double x_BI = 0, y_BI = 0, z_BI = 0; // body intersection point coordinates for IB faces
		double rho_BI = 0; // pressure at body intersection point for IB faces
		double u_BI = 0;   // velocity u at body intersection point for IB faces
		double v_BI = 0;   // velocity v at body intersection point for IB faces
		double w_BI = 0;   // velocity w at body intersection point for IB faces
		double E_BI = 0;   // energy at body intersection point for IB faces
		double p_BI = 0;   // pressure at body intersection point for IB faces
		std::vector<int> adjacentCells; // use for k-least squares
		std::vector<double> adjacentCellsCx; // x-coords of adjacent cells
		std::vector<double> adjacentCellsCy; // y-coords of adjacent cells
		kExactLeastSquare kls; // least squares object for this face

		// default constructor to initialize kls (kExactLeastSquare has no default ctor)
        Face()
            : n1(-1), n2(-1), leftCell(-1), rightCell(-1),
              cx(0), cy(0), cz(0), nx(0), ny(0), nz(0), area(0),
              isBoundary(false), isImmersedBoundary(false),
              ib_nx(0), ib_ny(0), ib_nz(0),
              x_mirror(0), y_mirror(0), z_mirror(0),
              x_BI(0), y_BI(0), z_BI(0),
              rho_BI(0), u_BI(0), v_BI(0), w_BI(0), E_BI(0), p_BI(0),
              adjacentCells(), adjacentCellsCx(), adjacentCellsCy(),
              kls(std::vector<double>{}, std::vector<double>{}, 0.0, 0.0)
        {}
	};

	std::vector<Face> faces; // all unique faces
	std::vector<std::array<int,4>> cell_nodes; // node indices (4 corners) per cell
	std::vector<std::vector<int>> cell_faces; // face indices per cell (4 faces)

	std::vector<int> farfieldFaces; // indices of farfield boundary faces
	std::vector<int> farfieldFacesX_m1; // indices of farfield faces in x-direction
	std::vector<int> farfieldFacesX_p1; // indices of farfield faces in x-direction
	std::vector<int> farfieldFacesY_m1; // indices of farfield faces in y-direction
	std::vector<int> farfieldFacesY_p1; // indices of farfield faces in y-direction
	std::vector<int> farfieldFacesZ_m1; // indices of farfield faces in z-direction
	std::vector<int> farfieldFacesZ_p1; // indices of farfield faces in z-direction

	std::vector<int> immersedBoundaryFaces; // indices of immersed boundary faces
	std::vector<int> ibFacesX_m1; // indices of immersed boundary faces in x-direction
	std::vector<int> ibFacesX_p1; // indices of immersed boundary faces in x-direction
	std::vector<int> ibFacesY_m1; // indices of immersed boundary faces in y-direction
	std::vector<int> ibFacesY_p1; // indices of immersed boundary faces in y-direction
	std::vector<int> ibFacesZ_m1; // indices of immersed boundary faces in z-direction
	std::vector<int> ibFacesZ_p1; // indices of immersed boundary faces in z-direction

	std::vector<int> fluidFaces; // indices of fluid domain faces without boundaries
	std::vector<int> fluidFacesX; // indices of fluid faces in x-direction
	std::vector<int> fluidFacesX_m1; // special case for calculating dissipation
	std::vector<int> fluidFacesX_p1; // special case for calculating dissipation
	std::vector<int> fluidFacesY; // indices of fluid faces in y-direction
	std::vector<int> fluidFacesY_m1; // special case for calculating dissipation
	std::vector<int> fluidFacesY_p1; // special case for calculating dissipation
	std::vector<int> fluidFacesZ; // indices of fluid faces in z-direction
	std::vector<int> fluidFacesZ_m1; // special case for calculating dissipation
	std::vector<int> fluidFacesZ_p1; // special case for calculating dissipation

	std::vector<int> fluidCells; // indices of fluid cells
	std::vector<int> ghostCells; // indices of ghost cells
	std::vector<double> ghostCells_BI_rho;
	std::vector<double> ghostCells_BI_u;
	std::vector<double> ghostCells_BI_v;
	std::vector<double> ghostCells_BI_E;
	std::vector<double> ghostCells_BI_p;
	std::vector<double> ghostCells_mirror_rho;
	std::vector<double> ghostCells_mirror_u;
	std::vector<double> ghostCells_mirror_v;
	std::vector<double> ghostCells_mirror_E;
	std::vector<double> ghostCells_mirror_p;
	std::vector<std::vector<int>> adjacentCells; // use for k-least squares
	std::vector<std::vector<double>> adjacentCellsCx; // x-coords of adjacent cells
	std::vector<std::vector<double>> adjacentCellsCy; // y-coords of adjacent cells
	std::vector<kExactLeastSquare> kls; // least squares object for this face
	std::vector<double> ghostCells_ib_nx;
	std::vector<double> ghostCells_ib_ny;
	std::vector<double> ghostCells_ib_nz;
	std::vector<double> ghostCells_x_mirror;
	std::vector<double> ghostCells_y_mirror;
	std::vector<double> ghostCells_z_mirror;
	std::vector<double> ghostCells_x_BI;
	std::vector<double> ghostCells_y_BI;
	std::vector<double> ghostCells_z_BI;


	Mesh() = default;

	// Load a structured mesh from a CGNS file (reads node coordinates and ni/nj)
	// Returns true on success.
	bool loadFromCGNS(const std::string& filename);

	bool writeToCGNS(const std::string& filename);

	bool writeToCGNSWithCellData(const std::string& filename, const SpatialDiscretization& discretization);

	// Compute cell-centered metrics (centers, faces, normals, areas) for structured grid
	// Must be called after node coordinates and ni,nj are set.
	void computeStructuredMetrics();

	void levelSet(std::vector<double> geom_x, std::vector<double> geom_y);

	// compute normals for immersed boundary faces pointing from face center to body surface
	void computeImmersedBoundaryNormals(const std::vector<double>& geom_x, const std::vector<double>& geom_y);

	void assignFaceAndCellTypes();

	// helpers
	inline int nodeIndex(int i, int j) const { return j*ni + i; }
	inline int cellIndex(int ci, int cj) const { return cj*(ni-1) + ci; }
};
#endif // MESH_H