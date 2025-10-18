#include "mesh.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#include <map>
#include <cmath>

// CGNS C library
#include <cgnslib.h>

bool Mesh::loadFromCGNS(const std::string& filename) {
	std::ostringstream summary_out;

	int file_index;
	int ier = cg_open(filename.c_str(), CG_MODE_READ, &file_index);
	if (ier != CG_OK) {
		summary_out << "Failed to open CGNS file: " << cg_get_error();
		summary = summary_out.str();
		return false;
	}

	summary_out << "CGNS file: " << filename << "\n";

	int nbases = 0;
	cg_nbases(file_index, &nbases);
	summary_out << "Bases: " << nbases << "\n";

	if (nbases < 1) {
		cg_close(file_index);
		summary = summary_out.str();
		return false;
	}

	bool gotX=false, gotY=false;

	// scan bases/zones to find coordinate arrays (be permissive for geometry files)
	for (int ibase=1; ibase<=nbases && !(gotX && gotY); ++ibase) {
		int nzones = 0;
		cg_nzones(file_index, ibase, &nzones);
		for (int iz=1; iz<=nzones && !(gotX && gotY); ++iz) {
			char zonename[33];
			cgsize_t zsize[9];
			cg_zone_read(file_index, ibase, iz, zonename, zsize);
			summary_out << "Zone: " << zonename << "\n";

			int index_dim = 0;
			cg_index_dim(file_index, ibase, iz, &index_dim);

			// compute total number of coordinate points (product of zsize entries up to index_dim)
			size_t totalNodes = 1;
			for (int d=0; d<index_dim; ++d) {
				if (zsize[d] > 0) totalNodes *= (size_t)zsize[d];
			}
			if (totalNodes == 0) continue; // nothing to read

			// if structured 2D, set ni/nj for metric computations later
			if (index_dim >= 2 && zsize[0] > 1 && zsize[1] > 1) {
				ni = (int)zsize[0]; nj = (int)zsize[1];
				summary_out << "ni=" << ni << " nj=" << nj << "\n";
			} else {
				// treat as 1D list of points (geometry)
				ni = (int)totalNodes; nj = 1;
				summary_out << "treated as point-list, npoints=" << totalNodes << "\n";
			}

			int ncoords = 0;
			cg_ncoords(file_index, ibase, iz, &ncoords);
			for (int ic = 1; ic <= ncoords; ++ic) {
				char coordname[33];
				CGNS_ENUMT(DataType_t) dtype;
				cg_coord_info(file_index, ibase, iz, ic, &dtype, coordname);
				std::string nm(coordname);

				// prepare range arrays for cg_coord_read (3 elements expected)
				cgsize_t range_min[3] = {1,1,1};
				cgsize_t range_max[3] = {1,1,1};
				if (index_dim >= 1) range_max[0] = (cgsize_t) ((index_dim>=1)? zsize[0] : 1);
				if (index_dim >= 2) range_max[1] = (cgsize_t) zsize[1];
				if (index_dim >= 3) range_max[2] = (cgsize_t) zsize[2];

				std::vector<double> buf(totalNodes);
				int ier2 = cg_coord_read(file_index, ibase, iz, coordname, CGNS_ENUMV(RealDouble), range_min, range_max, buf.data());
				if (ier2 != CG_OK) {
					// continue to next coordinate
					continue;
				}

				if (nm.find("CoordinateX") != std::string::npos || nm.find("GridCoordinatesX") != std::string::npos || nm == "X") { x = std::move(buf); gotX=true; }
				else if (nm.find("CoordinateY") != std::string::npos || nm.find("GridCoordinatesY") != std::string::npos || nm == "Y") { y = std::move(buf); gotY=true; }
				else if (nm.find("CoordinateZ") != std::string::npos || nm.find("GridCoordinatesZ") != std::string::npos || nm == "Z") { z = std::move(buf); }
			}
		}
	}

	cg_close(file_index);
	summary = summary_out.str();

	// if we found structured 2D grid, prepare cell containers and compute metrics
	if (ni > 1 && nj > 1) {
		ncells = (ni-1)*(nj-1);
		cx.assign(ncells, 0.0);
		cy.assign(ncells, 0.0);
		cz.assign(ncells, 0.0);
		volume.assign(ncells, 0.0);
		cell_types.assign(ncells, 0);
		phi.assign(ncells, 0.0);
		computeStructuredMetrics();
	}

	// return success if at least X and Y were read
	return (gotX && gotY);


}


void Mesh::computeStructuredMetrics() {
	if (ni < 2 || nj < 2) return;

	// build cell_nodes and cell_faces
	ncells = (ni-1)*(nj-1);
	cell_nodes.resize(ncells);
	cell_faces.clear();
	faces.clear();
	cell_faces.resize(ncells);

	auto getNode = [&](int i, int j)->int { return nodeIndex(i,j); };

	// create cells and nodes
	for (int cj = 0; cj < nj-1; ++cj) {
		for (int ci = 0; ci < ni-1; ++ci) {
			int c = cellIndex(ci,cj);
			int n00 = getNode(ci, cj);
			int n10 = getNode(ci+1, cj);
			int n11 = getNode(ci+1, cj+1);
			int n01 = getNode(ci, cj+1);
			cell_nodes[c] = {n00, n10, n11, n01};
		}
	}

	// helper to add/get face between two nodes
	auto make_face_key = [&](int a, int b){ if (a<b) return std::pair<int,int>(a,b); else return std::pair<int,int>(b,a); };
	std::map<std::pair<int,int>, int> faceMap;

	for (int c=0; c<ncells; ++c) {
		auto &cn = cell_nodes[c];
		std::array<int,4> face_nodes = {cn[0], cn[1], cn[2], cn[3]};
		// four faces: (0-1), (1-2), (2-3), (3-0)
		std::array<std::pair<int,int>,4> fn = { make_face_key(face_nodes[0],face_nodes[1]), make_face_key(face_nodes[1],face_nodes[2]), make_face_key(face_nodes[2],face_nodes[3]), make_face_key(face_nodes[3],face_nodes[0]) };
		for (int f=0; f<4; ++f) {
			auto it = faceMap.find(fn[f]);
			int fid;
			if (it == faceMap.end()) {
				fid = (int)faces.size();
				faceMap[fn[f]] = fid;
				Face F; F.n1 = fn[f].first; F.n2 = fn[f].second; F.leftCell = c; F.rightCell = -1;
				// compute face center and normal now
				double x1 = x[F.n1], y1 = y[F.n1];
				double x2 = x[F.n2], y2 = y[F.n2];
				F.cx = 0.5*(x1 + x2);
				F.cy = 0.5*(y1 + y2);
				// 2D normal (nx,ny) outward from left to right edge (perp to edge vector)
				double tx = x2 - x1;
				double ty = y2 - y1;
				double length = std::sqrt(tx*tx + ty*ty);
				if (length > 0) {
					F.nx = ty/length; F.ny = -tx/length; F.area = length;
				} else { F.nx = 0; F.ny = 0; F.area = 0; }
				faces.push_back(F);
			} else {
				fid = it->second;
				// set right cell
				faces[fid].rightCell = c;
			}
			cell_faces[c].push_back(fid);
		}
	}

	// compute cell centers as average of node coords
	for (int cj = 0; cj < nj-1; ++cj) {
		for (int ci = 0; ci < ni-1; ++ci) {
			int c = cellIndex(ci,cj);
			auto &cn = cell_nodes[c];
			double sx=0, sy=0, sz=0;
			for (int k=0;k<4;++k) {
				int n = cn[k];
				sx += x[n]; sy += y[n]; if (!z.empty()) sz += z[n];
			}
			cx[c] = sx/4.0; cy[c] = sy/4.0; cz[c] = (z.empty()?0.0:sz/4.0);
		}
	}

	// compute cell volumes (areas in 2D) using shoelace formula
	for (int c=0; c<ncells; ++c) {
		auto &cn = cell_nodes[c];
		double area = 0.0;
		for (int k=0; k<4; ++k) {
			int n1 = cn[k];;
			int n2 = cn[(k+1)%4];
			area += x[n1]*y[n2] - x[n2]*y[n1];
		}
		volume[c] = 0.5 * std::abs(area);
	}

	// Post-process faces: set isBoundary and orient normals to point from leftCell -> rightCell
	for (size_t fid = 0; fid < faces.size(); ++fid) {
		auto &F = faces[fid];
		F.isBoundary = (F.leftCell < 0 || F.rightCell < 0);
		// If both adjacent cells exist, orient normal from left to right
		if (F.leftCell >= 0 && F.rightCell >= 0) {
			double vx = cx[F.rightCell] - cx[F.leftCell];
			double vy = cy[F.rightCell] - cy[F.leftCell];
			double dot = vx * F.nx + vy * F.ny;
			if (dot < 0) {
				// flip normal to make it point left->right
				F.nx = -F.nx; F.ny = -F.ny; F.nz = -F.nz;
				std::swap(F.n1, F.n2);
			}
		} else if (F.leftCell >= 0 && F.rightCell < 0) {
			// boundary face: ensure normal points outward from the cell (from cell center to face center)
			double vx = F.cx - cx[F.leftCell];
			double vy = F.cy - cy[F.leftCell];
			double dot = vx * F.nx + vy * F.ny;
			if (dot < 0) {
				F.nx = -F.nx; F.ny = -F.ny; F.nz = -F.nz;
				std::swap(F.n1, F.n2);
			}
		} else if (F.rightCell >= 0 && F.leftCell < 0) {
			// unlikely in current construction, but handle symmetrically: ensure normal points from left(-1) to rightCell
			double vx = cx[F.rightCell] - F.cx; // vector right cell center <- face center
			double vy = cy[F.rightCell] - F.cy;
			double dot = vx * F.nx + vy * F.ny;
			if (dot > 0) {
				// flip so that normal would point from left(empty) -> rightCell (i.e., towards rightCell)
				F.nx = -F.nx; F.ny = -F.ny; F.nz = -F.nz;
				std::swap(F.n1, F.n2);
			}
		}
	}
}

void Mesh::levelSet(std::vector<double> geom_x, std::vector<double> geom_y) {
	// Expect geom_x and geom_y to contain ordered boundary points forming a closed loop
	size_t ngeom = std::min(geom_x.size(), geom_y.size());
	if (ngeom < 2 || ncells <= 0) return;

	// build list of segments (A->B) with wrap-around
	struct Seg { double ax, ay, bx, by; };
	std::vector<Seg> segs;
	segs.reserve(ngeom);
	for (size_t k = 0; k < ngeom; ++k) {
		size_t nk = (k + 1) % ngeom;
		Seg s; s.ax = geom_x[k]; s.ay = geom_y[k]; s.bx = geom_x[nk]; s.by = geom_y[nk];
		segs.push_back(s);
	}

	// ensure phi and cell_types are allocated
	if ((int)phi.size() != ncells) phi.assign(ncells, 0.0);
	if ((int)cell_types.size() != ncells) cell_types.assign(ncells, 1);

	// helper: distance point to segment
	auto dist_point_segment = [](double px, double py, const Seg &s)->double {
		double ax = s.ax, ay = s.ay, bx = s.bx, by = s.by;
		double abx = bx - ax, aby = by - ay;
		double apx = px - ax, apy = py - ay;
		double ab2 = abx*abx + aby*aby;
		double t = 0.0;
		if (ab2 > 0.0) t = (apx*abx + apy*aby) / ab2;
		double cx, cy;
		if (t <= 0.0) { cx = ax; cy = ay; }
		else if (t >= 1.0) { cx = bx; cy = by; }
		else { cx = ax + t*abx; cy = ay + t*aby; }
		double dx = px - cx, dy = py - cy; return std::sqrt(dx*dx + dy*dy);
	};

	// helper: test horizontal ray (to +x) intersection with segment
	auto ray_intersects = [](double px, double py, const Seg &s)->bool {
		double ay = s.ay, by = s.by, ax = s.ax, bx = s.bx;
		// exclude horizontal segments
		if (ay == by) return false;
		// standard test: does the ray at y=py cross the segment vertically?
		bool cond = ( (ay > py) != (by > py) );
		if (!cond) return false;
		// compute intersection x coordinate
		double xint = ax + (py - ay) * (bx - ax) / (by - ay);
		return xint > px;
	};

	// 2. Loop over all cell centers
	for (int c = 0; c < ncells; ++c) {
		double px = cx[c];
		double py = cy[c];

		// (a) unsigned distance: min distance to any segment
		double min_dist = std::numeric_limits<double>::infinity();
		for (const auto &s : segs) {
			double d = dist_point_segment(px, py, s);
			if (d < min_dist) min_dist = d;
		}

		// (b) determine sign using ray casting
		int intersections = 0;
		for (const auto &s : segs) if (ray_intersects(px, py, s)) ++intersections;

		int sign = (intersections % 2 == 0) ? 1 : -1; // even -> outside/fluid, odd -> inside/solid

		phi[c] = sign * min_dist;
	}

	// 3. classify cells: phi>0 => FLUID (1), else SOLID (-1)
	for (int c = 0; c < ncells; ++c) {
		if (phi[c] > 0.0) cell_types[c] = 1;
		else cell_types[c] = -1;
	}

	// 4. identify ghost cells: solid cells adjacent to any fluid cell become GHOST (0)
	int ncx = ni - 1; // cells in i-direction
	int ncy = nj - 1; // cells in j-direction
	for (int c = 0; c < ncells; ++c) {
		if (cell_types[c] != -1) continue; // only consider solid cells
		int ci = c % ncx;
		int cj = c / ncx;
		bool adjacentFluid = false;
		// neighbor offsets: left,right,down,up
		const int di[4] = {-1,1,0,0};
		const int dj[4] = {0,0,-1,1};
		for (int k=0;k<4 && !adjacentFluid;++k) {
			int nii = ci + di[k];
			int njj = cj + dj[k];
			if (nii < 0 || nii >= ncx || njj < 0 || njj >= ncy) continue;
			int nc = cellIndex(nii, njj);
			if (nc >= 0 && nc < ncells && cell_types[nc] == 1) adjacentFluid = true;
		}
		if (adjacentFluid) cell_types[c] = 0; // ghost
	}

	// 5. identify ghost cells immersed boundary faces: faces between fluid and solid cells
	for (size_t fid = 0; fid < faces.size(); ++fid) {
		auto &F = faces[fid];
		if (F.leftCell < 0 || F.rightCell < 0) continue;
		int leftType = cell_types[F.leftCell];;
		int rightType = cell_types[F.rightCell];
		if ( (leftType == 1 && rightType == 0) || (leftType == 0 && rightType == 1) ) {
			// face between fluid and solid cell: mark as immersed boundary
			F.isImmersedBoundary = true;
		}
	}
}

void Mesh::assignFaceTypes() {
	farfieldFaces.clear();
	immersedBoundaryFaces.clear();
	fluidFaces.clear();

	for (size_t fid = 0; fid < faces.size(); ++fid) {
		auto &F = faces[fid];
		if (F.isImmersedBoundary) {
			immersedBoundaryFaces.push_back((int)fid);
		} else if (F.isBoundary) {
			farfieldFaces.push_back((int)fid);
		} else {
			int leftType = cell_types[F.leftCell];
			int rightType = cell_types[F.rightCell];
			if (leftType == 1 && rightType == 1) {
				fluidFaces.push_back((int)fid);
			}
		}
	}
}

// compute normals for immersed-boundary faces: vector from face center to nearest point on geometry
void Mesh::computeImmersedBoundaryNormals(const std::vector<double>& geom_x, const std::vector<double>& geom_y) {
	size_t ngeom = std::min(geom_x.size(), geom_y.size());
	if (ngeom < 2) return;

	struct Seg { double ax, ay, bx, by; };
	std::vector<Seg> segs; segs.reserve(ngeom);
	for (size_t k=0;k<ngeom;++k) {
		size_t nk = (k+1)%ngeom;
		segs.push_back({geom_x[k], geom_y[k], geom_x[nk], geom_y[nk]});
	}

	auto project_to_seg = [&](double px, double py, const Seg &s, double &cxp, double &cyp)->double {
		double ax = s.ax, ay = s.ay, bx = s.bx, by = s.by;
		double abx = bx - ax, aby = by - ay;
		double apx = px - ax, apy = py - ay;
		double ab2 = abx*abx + aby*aby;
		double t = 0.0;
		if (ab2 > 0.0) t = (apx*abx + apy*aby) / ab2;
		if (t <= 0.0) { cxp = ax; cyp = ay; }
		else if (t >= 1.0) { cxp = bx; cyp = by; }
		else { cxp = ax + t*abx; cyp = ay + t*aby; }
		double dx = px - cxp; double dy = py - cyp; return std::sqrt(dx*dx + dy*dy);
	};

	for (int fid : immersedBoundaryFaces) {
		if (fid < 0 || fid >= (int)faces.size()) continue;
		auto &F = faces[fid];
		// prefer the adjacent fluid cell center as the origin so the normal points from fluid -> body
		double ox = F.cx, oy = F.cy;
		int fluidCell = -1;
		if (F.leftCell >= 0 && cell_types[F.leftCell] == 1) fluidCell = F.leftCell;
		else if (F.rightCell >= 0 && cell_types[F.rightCell] == 1) fluidCell = F.rightCell;
		if (fluidCell >= 0) { ox = cx[fluidCell]; oy = cy[fluidCell]; }

		double min_d = std::numeric_limits<double>::infinity();
		double best_x=ox, best_y=oy;
		for (const auto &s : segs) {
			double cxp, cyp;
			double d = project_to_seg(ox, oy, s, cxp, cyp);
			if (d < min_d) { min_d = d; best_x = cxp; best_y = cyp; }
		}
		double vx = best_x - ox; double vy = best_y - oy;
		double norm = std::sqrt(vx*vx + vy*vy);
		if (norm > 0.0) { F.ib_nx = vx / norm; F.ib_ny = vy / norm; F.ib_nz = 0.0; }
		else { F.ib_nx = 0.0; F.ib_ny = 0.0; F.ib_nz = 0.0; }
	}
}

bool Mesh::writeToCGNS(const std::string& filename) {
	// Remove existing file to ensure a clean write
	std::remove(filename.c_str());
	// Open file for writing (create new)
	int file_index = 0;
	int ier = cg_open(filename.c_str(), CG_MODE_WRITE, &file_index);
	if (ier != CG_OK) {
		std::cerr << "cg_open write failed: " << cg_get_error() << std::endl;
		return false;
	}

	// choose cell/physical dims
	int cellDim = 2;
	int physDim = (z.empty() ? 2 : 3);

	int base = 0;
	ier = cg_base_write(file_index, "Base", cellDim, physDim, &base);
	if (ier != CG_OK) {
		std::cerr << "cg_base_write failed: " << cg_get_error() << std::endl;
		cg_close(file_index);
		return false;
	}

	// zone size: structured grid with ni x nj vertices
	int zone = 0;
	// For 2D structured grids CGNS commonly expects a 6-element size array: vertex_size(ni,nj,?), cell_size(ni-1,nj-1,?),
	// we'll provide the 2D-friendly 6-element variant (vertex then cell sizes) to avoid mismatches.
	// ordering: VertexSize[0..1], CellSize[0..1], BndSize[0..1]
	cgsize_t zone_size6[6] = { (cgsize_t)ni, (cgsize_t)nj, (cgsize_t)std::max(0, ni-1), (cgsize_t)std::max(0, nj-1), (cgsize_t)0, (cgsize_t)0 };
	ier = cg_zone_write(file_index, base, "Zone", zone_size6, CGNS_ENUMV(Structured), &zone);
	if (ier != CG_OK) {
		const char* msg = cg_get_error();
		if (msg && std::strlen(msg) > 0) std::cerr << "cg_zone_write failed: " << msg << std::endl;
		else std::cerr << "cg_zone_write failed with code " << ier << std::endl;
		cg_close(file_index);
		return false;
	}

	// write coordinates (assume node ordering matches mesh.x/mesh.y arrays)
	if (!x.empty()) {
		int coord_index = 0;
		int ret = cg_coord_write(file_index, base, zone, CGNS_ENUMV(RealDouble), "CoordinateX", x.data(), &coord_index);
		if (ret != CG_OK) {
			const char* msg = cg_get_error();
			std::cerr << "cg_coord_write X failed (ret=" << ret << "): " << (msg?msg:"(no message)") << std::endl;
		}
	}
	if (!y.empty()) {
		int coord_index = 0;
		int ret = cg_coord_write(file_index, base, zone, CGNS_ENUMV(RealDouble), "CoordinateY", y.data(), &coord_index);
		if (ret != CG_OK) {
			const char* msg = cg_get_error();
			std::cerr << "cg_coord_write Y failed (ret=" << ret << "): " << (msg?msg:"(no message)") << std::endl;
		}
	}
	if (!z.empty()) {
		int coord_index = 0;
		int ret = cg_coord_write(file_index, base, zone, CGNS_ENUMV(RealDouble), "CoordinateZ", z.data(), &coord_index);
		if (ret != CG_OK) {
			const char* msg = cg_get_error();
			std::cerr << "cg_coord_write Z failed (ret=" << ret << "): " << (msg?msg:"(no message)") << std::endl;
		}
	}

	// Create a solution node at cell centers and write cell-centered fields
	int sol = 0;
	ier = cg_sol_write(file_index, base, zone, "CellData", CGNS_ENUMV(CellCenter), &sol);
	if (ier != CG_OK) {
		std::cerr << "cg_sol_write failed: " << cg_get_error() << std::endl;
		cg_close(file_index);
		return false;
	}

	// Utility lambda to write a field if data exists and has correct size
	auto writeField = [&](const char* name, const double* data, size_t n)->void {
		if (!data) return;
		if ((int)n != ncells) {
			std::cerr << "Warning: field '" << name << "' size mismatch (expected " << ncells << ")\n";
			return;
		}
		int field_index = 0;
		int ret = cg_field_write(file_index, base, zone, sol, CGNS_ENUMV(RealDouble), name, (void*)data, &field_index);
		if (ret != CG_OK) {
			const char* msg = cg_get_error();
			std::cerr << "cg_field_write " << name << " failed (ret=" << ret << "): " << (msg?msg:"(no message)") << std::endl;
		}
	};

	// write double arrays
	if (!cx.empty()) writeField("cx", cx.data(), cx.size());
	if (!cy.empty()) writeField("cy", cy.data(), cy.size());
	if (!cz.empty()) writeField("cz", cz.data(), cz.size());
	if (!volume.empty()) writeField("volume", volume.data(), volume.size());
	if (!phi.empty()) writeField("phi", phi.data(), phi.size());

	// write integer cell_types as doubles (CGNS fields are typed; write as double conversion)
	if (!cell_types.empty()) {
		std::vector<double> ct_double(ncells);
		for (int i=0;i<ncells;++i) ct_double[i] = static_cast<double>(cell_types[i]);
		writeField("cell_types", ct_double.data(), ct_double.size());
	}

	// close file
	ier = cg_close(file_index);
	if (ier != CG_OK) {
		std::cerr << "cg_close failed: " << cg_get_error() << std::endl;
		return false;
	}
	return true;
}

