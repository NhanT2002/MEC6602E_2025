import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import read_PLOT3D_mesh, read_plot3d_2d, compute_coeff, read_residual_history


# increase font size for plots
plt.rcParams.update({'font.size': 14})

# Incompressible cases
MESHES = ["naca2412", "sc20712"]
for mesh in MESHES :
    data_euler = pd.read_csv(f"../polars/{mesh}_polar_mach_0.200.csv")
    data_hspm = pd.read_csv(f"../HSPM/polars/{mesh}_polars_results.csv")

    if mesh == "naca2412" :
        alpha_max = 8.544
        cl_max = 1.282
    elif mesh == "sc20712" :
        alpha_max = 8.62
        cl_max = 1.0681

    plt.figure()
    plt.plot(data_euler['aoa'], data_euler['cl'], 'o-', label='Euler Solver')
    plt.plot(data_hspm['aoa'], data_hspm['cl'], 's--', label='HSPM Solver')
    plt.plot(alpha_max, cl_max, 'r*', markersize=12, label='Valarezo $C_{l,max}$')
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("$C_l$")
    plt.title(f"Lift Coefficient vs Angle of Attack for {mesh.upper()}")
    plt.legend()
    plt.grid()
    plt.savefig(f"fig/M02/{mesh}_cl_comparison.pdf")

    plt.figure()
    plt.plot(data_euler['aoa'], data_euler['cd'], 'o-', label='Euler Solver')
    plt.plot(data_hspm['aoa'], data_hspm['cd'], 's--', label='HSPM Solver')
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("$C_d$")
    plt.title(f"Drag Coefficient vs Angle of Attack for {mesh.upper()}")
    plt.legend()
    plt.grid()
    plt.savefig(f"fig/M02/{mesh}_cd_comparison.pdf")

    plt.figure()
    plt.plot(data_euler['aoa'], data_euler['cm'], 'o-', label='Euler Solver')
    plt.plot(data_hspm['aoa'], data_hspm['cm'], 's--', label='HSPM Solver')
    plt.xlabel("Angle of Attack (degrees)")
    plt.ylabel("$C_m$")
    plt.title(f"Moment Coefficient vs Angle of Attack for {mesh.upper()}")
    plt.legend()
    plt.grid()
    plt.savefig(f"fig/M02/{mesh}_cm_comparison.pdf")

for mesh in MESHES :
    for alpha in [-5.0, 0.0, 5.0, 10.0, 15.0] :
        alpha_str = f"{alpha:.3f}"
        alpha_str_hspm = f"{alpha:.2f}"
        x, y = read_PLOT3D_mesh(f"/home/apollon/hitra2/MEC6602E/HW3/mesh/{mesh}.xyz")
        ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(f"/home/apollon/hitra2/MEC6602E/HW3/solver_outputs/{mesh}/mach_0.200/aoa_{alpha_str}/output_aoa_{alpha_str}.q")
        cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
        data_hspp = pd.read_csv(f"/home/apollon/hitra2/MEC6602E/HW3/HSPM/output/{mesh}/CPsol_A{alpha_str_hspm}.dat", skiprows=1, sep='\s+', names=['x', 'y', 'z', 'cp'])
        plt.figure()
        plt.plot(x[0], cp_airfoil, '-', label='Euler Solver')
        plt.plot(data_hspp['x'], data_hspp['cp'], '--', label='HSPM Solver')
        plt.gca().invert_yaxis()
        plt.xlabel("x")
        plt.ylabel("$C_p$")
        plt.title(f"$C_p$ distribution at $\\alpha$={alpha_str} deg for {mesh.upper()}")
        plt.legend()
        plt.grid()
        plt.savefig(f"fig/M02/{mesh}_cp_distribution_alpha_{alpha_str}.pdf")