#include "cgnsReader.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

// CGNS C library
#include <cgnslib.h>

// Return coordinate arrays and a short summary for the mesh file.
Mesh readMesh(const std::string& filename) {
	Mesh m;
	std::ostringstream summary;

	int file_index;
	int ier = cg_open(filename.c_str(), CG_MODE_READ, &file_index);
	if (ier != CG_OK) {
		summary << "Failed to open CGNS file: " << cg_get_error();
		m.summary = summary.str();
		return m;
	}

	summary << "CGNS file: " << filename << "\n";

	int nbases = 0;
	cg_nbases(file_index, &nbases);
	summary << "Bases: " << nbases << "\n";

	for (int ibase = 1; ibase <= nbases; ++ibase) {
		char basename[33];
		int cell_dim, phy_dim;
		cg_base_read(file_index, ibase, basename, &cell_dim, &phy_dim);
		summary << "  " << basename << " (cell_dim=" << cell_dim << ", phy_dim=" << phy_dim << ")\n";

		int nzones = 0;
		cg_nzones(file_index, ibase, &nzones);
		for (int izone = 1; izone <= nzones; ++izone) {
			char zonename[33];
			cgsize_t zsize[9];
			cg_zone_read(file_index, ibase, izone, zonename, zsize);
			summary << "    Zone " << izone << ": " << zonename << "\n";

			// Read coordinates (if present)
			int ncoords = 0;
			cg_ncoords(file_index, ibase, izone, &ncoords);
			summary << "      ncoords=" << ncoords << "\n";
			for (int ic = 1; ic <= ncoords; ++ic) {
				char coordname[33];
				CGNS_ENUMT(DataType_t) dtype;
				cg_coord_info(file_index, ibase, izone, ic, &dtype, coordname);
				summary << "        coord " << ic << ": " << coordname << "\n";

				// read the coordinate data
				// determine number of points from zsize for structured zones
				cgsize_t ni = 1;
				int index_dim = 0;
				cg_index_dim(file_index, ibase, izone, &index_dim);
				if (index_dim > 0) {
					// structured: product of zsize[0..index_dim-1]
					ni = 1;
					for (int d = 0; d < index_dim; ++d) ni *= zsize[d];
				} else {
					// unstructured: try reading elements count; fallback to 0
					ni = 0;
				}

				if (ni > 0) {
					std::vector<double> buf(ni);
					cg_coord_read(file_index, ibase, izone, coordname, CGNS_ENUMV(RealDouble), NULL, NULL, buf.data());
					// store into mesh.x/y/z depending on coord name
					std::string nm(coordname);
					if (nm.find("CoordinateX") != std::string::npos || nm == "CoordinateX" || nm == "CoordinateX") {
						m.x = std::move(buf);
					} else if (nm.find("CoordinateY") != std::string::npos || nm == "CoordinateY") {
						m.y = std::move(buf);
					} else if (nm.find("CoordinateZ") != std::string::npos || nm == "CoordinateZ") {
						m.z = std::move(buf);
					} else {
						// assign by position if x/y/z empty
						if (m.x.empty()) m.x = std::move(buf);
						else if (m.y.empty()) m.y = std::move(buf);
						else if (m.z.empty()) m.z = std::move(buf);
					}
				}
			}
		}
	}

	cg_close(file_index);
	m.summary = summary.str();
	return m;
}

// For geometry we return a small summary of the BC/Family/Location info
Mesh readGeometry(const std::string& filename) {
	Mesh m;
	std::ostringstream summary;

	int file_index;
	int ier = cg_open(filename.c_str(), CG_MODE_READ, &file_index);
	if (ier != CG_OK) {
		summary << "Failed to open CGNS file: " << cg_get_error();
		m.summary = summary.str();
		return m;
	}

	summary << "Geometry (CGNS): " << filename << "\n";

	int nbases = 0;
	cg_nbases(file_index, &nbases);
	for (int ibase = 1; ibase <= nbases; ++ibase) {
		char basename[33];
		int cell_dim, phy_dim;
		cg_base_read(file_index, ibase, basename, &cell_dim, &phy_dim);
		summary << "Base " << ibase << ": " << basename << "\n";

		int nz = 0;
		cg_nzones(file_index, ibase, &nz);
		for (int iz = 1; iz <= nz; ++iz) {
			char zonename[33];
			cgsize_t zsize[9];
			cg_zone_read(file_index, ibase, iz, zonename, zsize);
			summary << " Zone " << iz << ": " << zonename << "\n";

			// read coords similar to readMesh
			int ncoords = 0;
			cg_ncoords(file_index, ibase, iz, &ncoords);
			for (int ic = 1; ic <= ncoords; ++ic) {
				char coordname[33];
				CGNS_ENUMT(DataType_t) dtype;
				cg_coord_info(file_index, ibase, iz, ic, &dtype, coordname);

				cgsize_t ni = 1;
				int index_dim = 0;
				cg_index_dim(file_index, ibase, iz, &index_dim);
				if (index_dim > 0) {
					ni = 1;
					for (int d = 0; d < index_dim; ++d) ni *= zsize[d];
				} else ni = 0;

				if (ni > 0) {
					std::vector<double> buf(ni);
					cg_coord_read(file_index, ibase, iz, coordname, CGNS_ENUMV(RealDouble), NULL, NULL, buf.data());
					std::string nm(coordname);
					if (nm.find("CoordinateX") != std::string::npos) m.x = std::move(buf);
					else if (nm.find("CoordinateY") != std::string::npos) m.y = std::move(buf);
					else if (nm.find("CoordinateZ") != std::string::npos) m.z = std::move(buf);
				}
			}
		}
	}

	cg_close(file_index);
	m.summary = summary.str();
	return m;
}

