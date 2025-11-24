import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import read_PLOT3D_mesh, read_plot3d_2d, compute_coeff, read_residual_history


# increase font size for plots
plt.rcParams.update({'font.size': 14})

# Compressible cases
MESHES = ["naca2412", "sc20712"]
for mesh in MESHES :
    data_euler = pd.read_csv(f"../polars/{mesh}_polar_mach_0.500.csv")
    data_hspm = pd.read_csv(f"../HSPM/polars/{mesh}_polars_results.csv")

    cl_hspm_corrected = data_hspm['cl'] / np.sqrt(1 - 0.5**2)
    cm_hspm_corrected = data_hspm['cm'] / np.sqrt(1 - 0.5**2)

    plt.figure()
    plt.plot(data_euler['aoa'], data_euler['cl'], 'o-', label='Euler Solver')
    plt.plot(data_hspm['aoa'], data_hspm['cl'], 's--', label='HSPM Solver')
    plt.plot(data_hspm['aoa'], cl_hspm_corrected, 'd-.', label='HSPM Solver with \nPrandtl-Glauert Correction')
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("$C_l$")
    plt.title(f"Lift Coefficient vs Angle of Attack for {mesh.upper()}")
    plt.legend()
    plt.grid()
    plt.savefig(f"fig/M05/{mesh}_cl_comparison.pdf")

    plt.figure()
    plt.plot(data_euler['aoa'], data_euler['cm'], 'o-', label='Euler Solver')
    plt.plot(data_hspm['aoa'], data_hspm['cm'], 's--', label='HSPM Solver')
    plt.plot(data_hspm['aoa'], cm_hspm_corrected, 'd-.', label='HSPM Solver with \nPrandtl-Glauert Correction')
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("$C_m$")
    plt.title(f"Moment Coefficient vs Angle of Attack for {mesh.upper()}")
    plt.legend()
    plt.grid()
    plt.savefig(f"fig/M05/{mesh}_cm_comparison.pdf")

for mesh in MESHES :
    for alpha in [-5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] :
        alpha_str = f"{alpha:.3f}"
        alpha_str_hspm = f"{alpha:.2f}"
        x, y = read_PLOT3D_mesh(f"/home/apollon/hitra2/MEC6602E/HW3/mesh/{mesh}.xyz")
        ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(f"/home/apollon/hitra2/MEC6602E/HW3/solver_outputs/{mesh}/mach_0.500/aoa_{alpha_str}/output_aoa_{alpha_str}.q")
        cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
        data_hspp = pd.read_csv(f"/home/apollon/hitra2/MEC6602E/HW3/HSPM/output/{mesh}/CPsol_A{alpha_str_hspm}.dat", skiprows=1, sep='\s+', names=['x', 'y', 'z', 'cp'])
        
        cp_hspm_corrected = data_hspp['cp'] / np.sqrt(1 - 0.5**2)
        
        plt.figure()
        plt.plot(x[0], cp_airfoil, '-', label='Euler Solver')
        plt.plot(data_hspp['x'], data_hspp['cp'], '--', label='HSPM Solver')
        plt.plot(data_hspp['x'], cp_hspm_corrected, '-.', label='HSPM Solver with \nPrandtl-Glauert Correction')
        plt.gca().invert_yaxis()
        plt.xlabel("x")
        plt.ylabel("$C_p$")
        plt.title(f"$C_p$ distribution at $\\alpha$={alpha_str} deg for {mesh.upper()}")
        plt.legend()
        plt.grid()
        plt.savefig(f"fig/M05/{mesh}_cp_distribution_alpha_{alpha_str}.pdf")