import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from helper import read_PLOT3D_mesh, read_plot3d_2d, compute_coeff, read_residual_history, mach_airfoil

MESHES = ["naca2412", "sc20712", "crm"]

for mesh in MESHES :
    x, y = read_PLOT3D_mesh(f"/home/apollon/hitra2/MEC6602E/HW3/mesh/{mesh}.xyz")
    for alpha in np.arange(-5.0, 16.0, 0.5) :
        alpha_str = f"{alpha:.3f}"
        output_path = f"/home/apollon/hitra2/MEC6602E/HW3/solver_outputs/{mesh}/mach_0.700/aoa_{alpha_str}/output_aoa_{alpha_str}.q"
        # check if file exists
        if not os.path.isfile(output_path):
            print(f"File {output_path} does not exist. Skipping.")
            continue
        ni, nj, mach, alpha_rad, reyn, time, q_vertex = read_plot3d_2d(output_path)
        cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha_rad, T_inf=300, p_inf=1E5)
        mach_distribution = mach_airfoil(x, y, q_vertex, mach, alpha_rad, T_inf=300, p_inf=1E5)

        if np.max(mach_distribution) > 1.3 :
            plt.figure()
            plt.plot(x[0], mach_distribution)
            plt.axhline(y=1.3, color='r', linestyle='--', label='Mach 1.3')
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("Mach Number")
            plt.grid()
            plt.title(f"Mach distribution at $\\alpha$={alpha_str} deg for {mesh.upper()}")
            plt.savefig(f"fig/M07/{mesh}_buffet_mach_distribution_alpha_{alpha_str}.pdf")
            plt.close()