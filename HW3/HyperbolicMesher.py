import numpy as np
import sys  # Added for exit on error
from scipy.interpolate import interp1d

def naca4_generator(naca_code: str, num_points: int, cosine_spacing: bool = True):
    """
    Generates the coordinates of a NACA 4-digit airfoil.

    Args:
        naca_code (str): The 4-digit NACA code (e.g., '2412').
        num_points (int): The total number of points to define the airfoil surface.
                          Should be an odd number to have a point at the leading edge.
        cosine_spacing (bool): If True, clusters points at the leading and trailing edges.

    Returns:
        np.ndarray: An array of shape (num_points, 2) with the [x, y] coordinates
                    ordered from trailing edge (lower), around the leading edge,
                    to the trailing edge (upper).
    """
    if len(naca_code) != 4:
        raise ValueError("NACA code must be a 4-digit string.")
    if num_points % 2 == 0:
        print("Warning: num_points should be odd for a point at the leading edge. Incrementing by 1.")
        num_points += 1

    m = float(naca_code[0]) / 100.0
    p = float(naca_code[1]) / 10.0
    t = float(naca_code[2:]) / 100.0

    # Number of points for one side (upper or lower)
    n_side = (num_points + 1) // 2

    if cosine_spacing:
        beta = np.linspace(0, np.pi, n_side)
        x = 0.5 * (1 - np.cos(beta))
    else:
        x = np.linspace(0, 1, n_side)

    # Thickness distribution (modified for closed trailing edge)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

    # Camber line and its gradient
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if p > 0:
        front_mask = x < p
        back_mask = x >= p

        # Front section (0 <= x < p)
        yc[front_mask] = (m / p**2) * (2 * p * x[front_mask] - x[front_mask]**2)
        dyc_dx[front_mask] = (2 * m / p**2) * (p - x[front_mask])

        # Back section (p <= x <= 1)
        yc[back_mask] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[back_mask] - x[back_mask]**2)
        dyc_dx[back_mask] = (2 * m / (1 - p)**2) * (p - x[back_mask])

    theta = np.arctan(dyc_dx)

    # Upper surface coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    # Lower surface coordinates
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Combine and order for O-grid: TE -> LE (lower) then LE -> TE (upper)
    X = np.concatenate((np.flip(xl), xu[1:]))
    Y = np.concatenate((np.flip(yl), yu[1:]))

    return np.vstack((X, Y)).T


def naca5_generator(naca_code: str, num_points: int, cosine_spacing: bool = True):
    """
    Generates the coordinates of a standard NACA 5-digit airfoil (e.g., '23012').
    This function only supports standard camber (e.g., '2x0xx').
    Reflex camber (e.g., '2x1xx') is not supported.

    Args:
        naca_code (str): The 5-digit NACA code (e.g., '23012').
        num_points (int): The total number of points to define the airfoil surface.
                          Should be an odd number.
        cosine_spacing (bool): If True, clusters points at the leading and trailing edges.

    Returns:
        np.ndarray: An array of shape (num_points, 2) with the [x, y] coordinates.
    """
    if len(naca_code) != 5:
        raise ValueError("NACA code must be a 5-digit string.")
    if num_points % 2 == 0:
        print("Warning: num_points should be odd for a point at the leading edge. Incrementing by 1.")
        num_points += 1

    # p = int(naca_code[1]) * 0.05
    q = int(naca_code[2])
    t = float(naca_code[3:]) / 100.0

    if q != 0:
        raise NotImplementedError("NACA 5-digit reflex camber (e.g., '23112') is not supported. Only standard '2x0xx' series.")

    # Pre-computed constants for the '2x0xx' series
    p_map = {
        1: (0.05, 0.0580, 361.4),
        2: (0.10, 0.1260, 51.64),
        3: (0.15, 0.2025, 15.957),
        4: (0.20, 0.2900, 6.643),
        5: (0.25, 0.3910, 3.230)
    }
    p_key = int(naca_code[1])
    if p_key not in p_map:
        raise ValueError(f"Invalid camber parameter 'P={p_key}'. Must be in [1, 2, 3, 4, 5].")

    p, m, k1 = p_map[p_key]

    n_side = (num_points + 1) // 2

    if cosine_spacing:
        beta = np.linspace(0, np.pi, n_side)
        x = 0.5 * (1 - np.cos(beta))
    else:
        x = np.linspace(0, 1, n_side)

    # Thickness distribution (same as 4-digit, closed TE)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

    # Camber line and its gradient
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    front_mask = x < m
    back_mask = x >= m

    # Front section (0 <= x < m)
    yc[front_mask] = (k1 / 6.0) * (x[front_mask]**3 - 3 * m * x[front_mask]**2 + m**2 * (3 - m) * x[front_mask])
    dyc_dx[front_mask] = (k1 / 6.0) * (3 * x[front_mask]**2 - 6 * m * x[front_mask] + m**2 * (3 - m))

    # Back section (m <= x <= 1)
    yc[back_mask] = (k1 * m**3 / 6.0) * (1 - x[back_mask])
    dyc_dx[back_mask] = -(k1 * m**3 / 6.0)

    theta = np.arctan(dyc_dx)

    # Upper surface coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    # Lower surface coordinates
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Combine and order
    X = np.concatenate((np.flip(xl), xu[1:]))
    Y = np.concatenate((np.flip(yl), yu[1:]))

    return np.vstack((X, Y)).T


def read_airfoil_file(filename: str, num_points: int, cosine_spacing: bool = True):
    """
    Reads an airfoil coordinate file (e.g., from UIUC database),
    re-panels it to the desired number of points, and orders it for the O-grid.

    Assumes a standard file format (like Lednicer or Siepmann) where:
    - Points are ordered from TE (upper) -> LE -> TE (lower).
    - The first line may be a header (it is skipped if it contains text).

    Args:
        filename (str): The path to the airfoil .dat file.
        num_points (int): The total number of points to re-panel the airfoil to.
        cosine_spacing (bool): If True, clusters points at LE and TE.

    Returns:
        np.ndarray: An array of shape (num_points, 2) with the [x, y] coordinates
                    ordered from TE (lower) -> LE -> TE (upper).
    """
    if num_points % 2 == 0:
        print("Warning: num_points should be odd for a point at the leading edge. Incrementing by 1.")
        num_points += 1

    try:
        # Try to load, skipping first row (potential header)
        coords = np.loadtxt(filename, skiprows=1)
    except ValueError:
        try:
            # If that fails, try loading with no skipped rows
            coords = np.loadtxt(filename, skiprows=0)
        except IOError:
            print(f"Error: Could not read airfoil file: {filename}")
            raise
        except Exception as e:
            print(f"Error: An error occurred while parsing {filename}: {e}")
            raise

    # Find the leading edge (minimum x-coordinate)
    le_index = np.argmin(coords[:, 0])

    def interpolate_surface(surface_coords, n_points, cos_space):

        n_side = (n_points + 1) // 2
        """Helper to interpolate one surface based on arc length."""
        # Calculate cumulative arc length
        distances = np.sqrt(np.sum(np.diff(surface_coords, axis=0)**2, axis=1))
        s_arc = np.insert(np.cumsum(distances), 0, 0)

        s_arc = s_arc - max(s_arc)/2.0
        if s_arc[-1] == 0: # Failsafe for a single point
            return np.tile(surface_coords[0], (n_points, 1))

        if cos_space:
            beta = np.linspace( 0.0, np.pi, n_side)
            s_frac = (np.cos(beta))
            s_new = s_frac * s_arc[-1]/2.0 + s_arc[-1]/2.0
            s_new = np.hstack((-s_new,np.flip(s_new[0:-1])))
        else:
            s_new = np.linspace(-s_arc[-1], s_arc[-1], n_points)

        

        # Interpolate x and y as functions of arc length
        x_new = np.zeros(2*n_side)
        y_new = np.zeros(2*n_side)

        spline=interp1d(s_arc, surface_coords[:, 0], kind='cubic')
        x_new = spline(s_new)
        spline=interp1d(s_arc, surface_coords[:, 1], kind='cubic')
        y_new = spline(s_new)

        return np.vstack((x_new, y_new)).T

    coords_new = interpolate_surface(coords, num_points, cosine_spacing)

    X = coords_new[:,0]#np.concatenate((lower_new_flipped[:, 0], upper_new[1:, 0]))
    Y = coords_new[:,1]#np.concatenate((lower_new_flipped[:, 1], upper_new[1:, 1]))


    # Add 4 cells on the TE if think TE
    if (X[0] == X[-1]) and (Y[0] != Y[-1]):

        thick_te_coord_x = np.zeros(len(X)+4)
        thick_te_coord_y = np.zeros(len(Y)+4)

        te_spacing_y = (Y[-1] - Y[0])/4.0
        te_spacing_x = (X[-1] - X[0])/4.0

        thick_te_coord_x[0] = X[0]+2.0*te_spacing_x
        thick_te_coord_x[1] = X[0]+1.0*te_spacing_x
        thick_te_coord_x[2:-2] = X
        thick_te_coord_x[-2] = X[-1]-1.0*te_spacing_x
        thick_te_coord_x[-1] = X[-1]-2.0*te_spacing_x

        thick_te_coord_y[0] = Y[0]+2.0*te_spacing_y
        thick_te_coord_y[1] = Y[0]+1.0*te_spacing_y
        thick_te_coord_y[2:-2] = Y
        thick_te_coord_y[-2] = Y[-1]-1.0*te_spacing_y
        thick_te_coord_y[-1] = Y[-1]-2.0*te_spacing_y


        final_coords = np.vstack((thick_te_coord_x, thick_te_coord_y)).T

    else:
        final_coords = np.vstack((X, Y)).T


    return final_coords


def generate_airfoil(naca_code: str, num_points: int, cosine_spacing: bool = True):
    """
    Selects the correct NACA generator based on the code length.
    """
    if len(naca_code) == 4:
        print(f"Generating NACA {naca_code} (4-digit) airfoil geometry...")
        return naca4_generator(naca_code, num_points, cosine_spacing)
    elif len(naca_code) == 5:
        print(f"Generating NACA {naca_code} (5-digit) airfoil geometry...")
        return naca5_generator(naca_code, num_points, cosine_spacing)
    else:
        raise ValueError(f"Invalid NACA code '{naca_code}'. Must be 4 or 5 digits.")


def hyperbolic_mesher(wall_points: np.ndarray, n_layers: int, initial_step: float, growth_factor: float,
                      smoothing_factor: float = 0.1, smoothing_iterations: int = 2):
    """
    Generates a 2D O-type hyperbolic mesh around a given airfoil geometry.

    The method marches outwards from the wall (J-min) to the farfield (J-max).
    It ensures the mesh closes properly at the trailing edge (TE).

    Args:
        wall_points (np.ndarray): Airfoil coordinates, shape (I_max, 2).
                                  Assumes points are ordered from TE-lower,
                                  around LE, to TE-upper.
                                  Assumes wall_points[0] and wall_points[-1]
                                  are the two TE points.
        n_layers (int): Number of layers to generate outwards (J_max).
        initial_step (float): The initial mesh spacing normal to the wall.
        growth_factor (float): The rate at which the normal spacing increases per layer.
        smoothing_factor (float): The strength of the explicit smoothing (0 to disable).
        smoothing_iterations (int): Number of smoothing passes per layer.

    Returns:
        np.ndarray: The generated mesh with shape (I_max, J_max, 2).
    """
    i_max = wall_points.shape[0]
    j_max = n_layers
    mesh = np.zeros((i_max, j_max, 2))

    # Set the wall boundary condition (J=0)
    mesh[:, 0, :] = wall_points

    # March outwards from J=0 to J=J_max-1
    for j in range(j_max - 1):
        # Current layer coordinates
        x_j = mesh[:, j, 0]
        y_j = mesh[:, j, 1]

        # Calculate tangent vectors using central differences.
        # np.roll handles the periodic connection at the trailing edge.
        x_plus_1 = np.roll(x_j, -1)
        x_minus_1 = np.roll(x_j, 1)
        y_plus_1 = np.roll(y_j, -1)
        y_minus_1 = np.roll(y_j, 1)

        tx = x_plus_1 - x_minus_1  # Tangent x-component
        ty = y_plus_1 - y_minus_1  # Tangent y-component

        # Calculate normal vector (rotate tangent by -90 degrees for outward pointing)
        nx = -ty
        ny = tx

        # Normalize the normal vector
        norm = np.sqrt(nx**2 + ny**2)
        # Add small epsilon to prevent division by zero (e.g., at sharp TE)
        norm = np.where(norm < 1e-12, 1e-12, norm)
        nx /= norm
        ny /= norm

        # --- Trailing Edge Handling ---
        # At the trailing edge (i=0 and i=i_max-1), we want the extrusion
        # to be along a single, shared vector, pointing straight back.

        # Find the bisector vector at the TE
        # Normal at TE-lower (i=0)
        nx_te_lower = nx[0]
        ny_te_lower = ny[0]
        # Normal at TE-upper (i=-1 or i_max-1)
        nx_te_upper = nx[-1]
        ny_te_upper = ny[-1]

        # Average the two normals to get a (mostly) x-aligned vector
        nx_te = 0.5 * (nx_te_lower + nx_te_upper)
        ny_te = 0.5 * (ny_te_lower + ny_te_upper)

        # Force it to be purely in the +x direction for a clean wake cut
        nx_te = 1.0
        ny_te = 0.0

        # Apply this vector to both TE points
        nx[0] = nx_te
        ny[0] = ny_te
        nx[-1] = nx_te
        ny[-1] = ny_te
        # -----------------------------

        # Average the nx between adjacent Cells for better treatment of the TE corners.
        for i in range(0,len(nx)-1):
            nx[i] = 0.5*(nx[i]+nx[i+1])
            nx[len(nx)-i-1] = 0.5*(nx[len(nx)-i-1]+nx[len(nx)-i-2])
            ny[i] = 0.5*(ny[i]+ny[i+1])
            ny[len(nx)-i-1] = 0.5*(ny[len(ny)-i-1]+ny[len(ny)-i-2])

        # Calculate the step distance for this layer
        step = initial_step * (growth_factor ** j)

        # Calculate the coordinates of the next layer
        x_j_plus_1 = x_j + step * nx
        y_j_plus_1 = y_j + step * ny

        # --- Force TE points to be identical ---
        # Average the new TE points to ensure they are at the *exact* same spot
        x_te_new_avg = 0.5 * (x_j_plus_1[0] + x_j_plus_1[-1])
        y_te_new_avg = 0.5 * (y_j_plus_1[0] + y_j_plus_1[-1])

        x_j_plus_1[0] = x_te_new_avg
        y_j_plus_1[0] = y_te_new_avg
        x_j_plus_1[-1] = x_te_new_avg
        y_j_plus_1[-1] = y_te_new_avg
        # ----------------------------------------

        # Apply explicit smoothing to the new layer to prevent grid crossing
        if j>1:
            if smoothing_factor > 0:
                for _ in range(smoothing_iterations):
                    # Use np.roll for periodic boundaries
                    x_prev = np.roll(x_j_plus_1, 1)
                    x_next = np.roll(x_j_plus_1, -1)
                    y_prev = np.roll(y_j_plus_1, 1)
                    y_next = np.roll(y_j_plus_1, -1)

                    # Laplacian smoothing
                    x_j_plus_1_smooth = x_j_plus_1 + smoothing_factor * (x_prev - 2 * x_j_plus_1 + x_next)
                    y_j_plus_1_smooth = y_j_plus_1 + smoothing_factor * (y_prev - 2 * y_j_plus_1 + y_next)

                    x_te_smooth_avg = 0.5 * (x_j_plus_1_smooth[0] + x_j_plus_1_smooth[-1])
                    y_te_smooth_avg = 0.5 * (y_j_plus_1_smooth[0] + y_j_plus_1_smooth[-1])

                    x_j_plus_1_smooth[0] = x_te_smooth_avg
                    y_j_plus_1_smooth[0] = y_te_smooth_avg
                    x_j_plus_1_smooth[-1] = x_te_smooth_avg
                    y_j_plus_1_smooth[-1] = y_te_smooth_avg

                    x_j_plus_1 = x_j_plus_1_smooth
                    y_j_plus_1 = y_j_plus_1_smooth

        # Store the new layer in the mesh
        mesh[:, j + 1, 0] = x_j_plus_1
        mesh[:, j + 1, 1] = y_j_plus_1

    return mesh


def elliptic_smoother(mesh: np.ndarray, iterations: int, omega: float):
    """
    Applies 2D elliptic smoothing (Laplacian) to the interior grid points
    using Jacobi iteration with successive over-relaxation (SOR).
    This smooths the grid in both I and J directions.
    The wall (j=0) and farfield (j=j_max-1) boundaries are held fixed.

    The strength of the smoothing is MODULATED: it is very weak near the
    wall (j=1) and ramps up to full strength at the farfield.

    Args:
        mesh (np.ndarray): The full mesh grid to be smoothed.
        iterations (int): The number of smoothing iterations to perform.
        omega (float): The successive over-relaxation factor (1.0 = standard Jacobi,
                         1.0 < omega < 2.0 = over-relaxed, > 1.0 is recommended).

    Returns:
        np.ndarray: The smoothed mesh.
    """
    i_max, j_max, _ = mesh.shape

    # We need a 'new_mesh' to store results of one full iteration (Jacobi style)
    mesh_new = np.copy(mesh)

    # Pre-calculate the normalization factor for the j-weight
    # j ranges from 1 to j_max-2. We want j=1 -> 0.0 and j=j_max-2 -> 1.0
    denominator = j_max - 3.0
    if denominator <= 1e-6: # Handle case of very few J-layers
        denominator = 10.0

    for _ in range(iterations):
        # Read from 'mesh', write to 'mesh_new'
        for j in range(1, j_max - 1): # Iterate over interior j-layers

            # Calculate the weighting factor for this j-layer
            # We use a quadratic ramp: w_j = ( (j-1) / (j_max-3) )^2
            # This makes the smoothing very weak near the wall and
            # ramps up quickly towards the farfield.
            j_norm = (j - 1.0) / denominator
            w_j = j_norm * j_norm
            w_j = min(w_j,1.0)
            # Modulate the omega factor by the weight
            # No smoothing at j=1 (w_j=0), full smoothing at j=j_max-2 (w_j=1)
            omega_j = omega * w_j

            # If omega_j is effectively zero, skip the calculation for this layer
            if omega_j < 1e-6:
                continue

            for i in range(i_max):     # Iterate over all i-points

                # Get i-neighbors (periodic)
                i_minus = (i - 1 + i_max) % i_max
                i_plus = (i + 1) % i_max

                # Get j-neighbors (non-periodic)
                j_minus = j - 1
                j_plus = j + 1

                # Get current old point from 'mesh'
                x_ij = mesh[i, j, 0]
                y_ij = mesh[i, j, 1]

                # 2D Laplacian average (target for the new point) from 'mesh'
                x_avg = 0.25 * (mesh[i_plus, j, 0] + mesh[i_minus, j, 0] + \
                                mesh[i, j_plus, 0] + mesh[i, j_minus, 0])
                y_avg = 0.25 * (mesh[i_plus, j, 1] + mesh[i_minus, j, 1] + \
                                mesh[i, j_plus, 1] + mesh[i, j_minus, 1])

                # Apply SOR with the *modulated* omega
                # x_new = x_old + omega_j * (x_target - x_old)
                x_new = x_ij + omega_j * (x_avg - x_ij)
                y_new = y_ij + omega_j * (y_avg - y_ij)

                # Store in the new mesh
                mesh_new[i, j, 0] = x_new
                mesh_new[i, j, 1] = y_new

        # --- Enforce TE closure on the new mesh ---
        # This ensures the wake cut points remain identical
        for j in range(1, j_max - 1):
            x_te_avg = 0.5 * (mesh_new[0, j, 0] + mesh_new[-1, j, 0])
            y_te_avg = 0.5 * (mesh_new[0, j, 1] + mesh_new[-1, j, 1])
            mesh_new[0, j, 0] = x_te_avg
            mesh_new[0, j, 1] = y_te_avg
            mesh_new[-1, j, 0] = x_te_avg
            mesh_new[-1, j, 1] = y_te_avg

        # After a full iteration, copy 'mesh_new' back to 'mesh' for the next read
        mesh = np.copy(mesh_new)

    return mesh # 'mesh' now contains the final smoothed result


def write_plot3d(mesh: np.ndarray, filename: str = "grid.xyz"):
    """
    Writes the 2D mesh to a 3D single-block, ASCII Plot3D file.
    The K-dimension (Z) is set to 1.

    Args:
        mesh (np.ndarray): The mesh to write, with shape (I, J, 2).
        filename (str): The output filename (e.g., 'grid.xyz').
    """
    i_max, j_max, _ = mesh.shape
    k_max = 1  # Single block in the Z-direction

    print(f"Writing Plot3D file: {filename}")
    print(f"Grid dimensions (I, J, K): ({i_max}, {j_max}, {k_max})")

    try:
        with open(filename, 'w') as f:

            # Write dimensions for block 1: i_max, j_max, k_max
            f.write(f"{i_max} {j_max}\n")

            # Create arrays for x, y, z coordinates
            x_coords = mesh[:, :, 0]
            y_coords = mesh[:, :, 1]
            z_coords = np.zeros((i_max, j_max)) # Z=0 for 2D

            # Write coordinates in Plot3D format (i-fastest, then j, then k)
            # This is Fortran-style flattening
            for j in range(j_max):
                for i in range(i_max):
                    f.write(f"{x_coords[i, j]:.10e} ")
                    f.write("\n")


            for j in range(j_max):
                for i in range(i_max):
                    f.write(f"{y_coords[i, j]:.10e} ")
                    f.write("\n")


        print(f"Successfully wrote {filename}")

    except IOError as e:
        print(f"Error: Could not write file {filename}: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # --- Mesh Generation Parameters ---

    # GEOMETRY_SOURCE:
    # - Use a 4 or 5-digit string (e.g., '2412', '23012') for NACA generators.
    # - Use a filename (e.g., 'naca835-216.dat') to read a coordinate file.
    GEOMETRY_SOURCE = 'mesh/sc20712.dat'  # <-- CHANGE THIS
    OUTPUT_FILENAME = "mesh/sc20712.xyz"
    # Example for NACA 5-series: GEOMETRY_SOURCE = '23012'
    # Example for NACA 8-series: GEOMETRY_SOURCE = 'naca835-216.dat'

    NUM_POINTS_ON_AIRFOIL = 125  # Number of points along the airfoil surface (I-direction)
    NUM_LAYERS = 129          # Number of layers marching away from the wall (J-direction)

    # --- Hyperbolic Marching Parameters ---
    INITIAL_STEP_SIZE = 0.01   # Initial distance of the first layer from the wall
    GROWTH_RATE = 1.06          # Growth factor for layer thickness (e.g., 1.04 = 4% growth per layer)

    # --- Smoothing Parameters ---
    SMOOTHING_FACTOR = 0.3    # How much to smooth each new layer (0=none, ~0.1-0.3 is typical)
    SMOOTHING_ITERATIONS = 2   # Number of smoothing passes per layer

    # --- Elliptic Smoothing Parameters ---
    DO_ELLIPTIC_SMOOTHING = true  # Set to True to run the elliptic smoother
    ELLIPTIA_ITERATIONS = 50      # Number of iterations for the elliptic solver
    ELLIPTIC_OMEGA = 1.0          # Over-relaxation factor (1.0 < omega < 2.0)

    # --- Output Parameters ---
    # Clean up geometry source string for filename
    base_name = GEOMETRY_SOURCE.replace('.dat', '').replace('.txt', '').replace('.', '_')


    # Generate the airfoil geometry
    try:
        if GEOMETRY_SOURCE.endswith('.dat') or GEOMETRY_SOURCE.endswith('.txt'):
            print(f"Reading geometry from file: {GEOMETRY_SOURCE}...")
            airfoil_wall = read_airfoil_file(GEOMETRY_SOURCE, NUM_POINTS_ON_AIRFOIL, cosine_spacing = True)
        else:
            airfoil_wall = generate_airfoil(GEOMETRY_SOURCE, NUM_POINTS_ON_AIRFOIL)
    except (ValueError, NotImplementedError, FileNotFoundError) as e:
        print(f"Error generating airfoil: {e}")
        sys.exit(1)

    if airfoil_wall is not None:
        print(f"Successfully generated/read geometry. Wall points: {airfoil_wall.shape[0]}")

    # Generate the hyperbolic mesh
    print("Generating hyperbolic mesh...")
    grid = hyperbolic_mesher(
        wall_points=airfoil_wall,
        n_layers=NUM_LAYERS,
        initial_step=INITIAL_STEP_SIZE,
        growth_factor=GROWTH_RATE,
        smoothing_factor=SMOOTHING_FACTOR,
        smoothing_iterations=SMOOTHING_ITERATIONS
    )
    print(f"Hyperbolic mesh generation complete. Mesh dimensions (I x J x XY): {grid.shape}")


    # Apply elliptic smoothing (optional)
    if grid is not None and DO_ELLIPTIC_SMOOTHING:
        print(f"Applying 2D elliptic smoothing ({ELLIPTIA_ITERATIONS} iterations with omega={ELLIPTIC_OMEGA})...")
        print("  (Smoothing strength is modulated from 0 at wall to full at farfield)")
        grid = elliptic_smoother(grid, ELLIPTIA_ITERATIONS, ELLIPTIC_OMEGA)
        print("Elliptic smoothing complete.")

    # Write the mesh to Plot3D format
    if grid is not None:
        write_plot3d(grid, OUTPUT_FILENAME)
