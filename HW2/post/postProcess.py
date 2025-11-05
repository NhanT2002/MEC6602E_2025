import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import read_PLOT3D_mesh, read_plot3d_2d, compute_coeff, read_residual_history

mesh_size = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# plt.figure()
# for size in mesh_size :
#     x, y = read_PLOT3D_mesh(f"../mesh/naca0012_{size}x{size}.xyz")
#     ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(f"../output/output_M08_A125_{size}.q")
#     cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
#     print(f"Mesh size: {size}x{size} => C_L: {C_L}, C_D: {C_D}, C_M: {C_M}")
#     plt.plot(x[0], cp_airfoil, label=f"{size}x{size}")

# plt.gca().invert_yaxis()
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("Cp")
# plt.title("NACA0012 Mach=0.8 alpha=1.25°")
# plt.grid()
# plt.tight_layout()

# Comparaison Vassberg-Jameson
ALPHA = {0.0 : "0", 1.25 : "125"}
MACH = {0.5 : "05", 0.8 : "08"}

x, y = read_PLOT3D_mesh("../mesh/naca0012_1024x1024.xyz")
for a in ALPHA.keys() : 
    for m in MACH.keys() :
        df = pd.read_csv(f"NACA0012_M{MACH[m]}_A{ALPHA[a]}.csv", header=None)
        plt.figure()
        if a == 0.0 :
            plt.plot(df[0], df[1], label="Vassberg-Jameson 4096x4096", color='tab:blue')
        else :
            plt.plot(df[0], df[1], label="Vassberg-Jameson 4096x4096", color='tab:blue')
            plt.plot(df[2], df[3], color='tab:blue')
        ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(f"../output/output_M{MACH[m]}_A{ALPHA[a]}_1024.q")
        cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
        plt.plot(x[0], cp_airfoil, '--', label="Euler Solver, k2 = 1/2, k4=1/64", color='tab:orange')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Cp")
        plt.title(f"NACA0012 Mach={m} alpha={a}°")
        plt.grid()
        plt.tight_layout()

# Residual history plotting
for a in ALPHA.keys() : 
    for m in MACH.keys() :
        plt.figure()
        for size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048] :
            Time, R0, R1, R2, R3, cl, cd, cm = read_residual_history(f"../residual_history/residual_history_M{MACH[m]}_A{ALPHA[a]}_{size}.txt")
            plt.semilogy(R0, label=f"{size}x{size}")
        plt.xlabel("Iterations")
        plt.ylabel("Residual continuity")
        plt.title(f"Residual history NACA0012 Mach={m} alpha={a}°")
        plt.grid()
        plt.legend()
        plt.tight_layout()


# Read results
results_M05_A0 = pd.read_csv("results_M05_A0.csv")
results_M05_A125 = pd.read_csv("results_M05_A125.csv")
results_M08_A0 = pd.read_csv("results_M08_A0.csv")
results_M08_A125 = pd.read_csv("results_M08_A125.csv")

# Plot convergence of CD for M=0.5, alpha=0°

def convergence_plot(results) :
    plt.figure()
    plt.loglog(1/results['Mesh Size'], results['C_D'], '-o')
    # add mesh size next to each point
    for i, size in enumerate(results['Mesh Size']) :
        plt.text(1/size, results['C_D'][i], f"{size}", fontsize=8, ha='right', va='bottom')
    plt.legend()
    plt.xlabel("1 / NC")
    plt.ylabel("$|C_D - C_D^*|$")
    plt.xlim(1/3000, 1/5)
    plt.grid()
    # compute the order of convergence and plot it
    CD_exact = 0
    errors = np.abs(results['C_D'][-3:] - CD_exact)
    hs = 1 / results['Mesh Size'][-3:]
    p = np.polyfit(np.log(hs), np.log(errors), 1)
    print(f"Order of convergence: {-p[0]:.2f}")
    plt.loglog(1/results['Mesh Size'], np.exp(p[1]) * (1/results['Mesh Size'])**p[0], '--', label=f"Order = {p[0]:.2f}")
    plt.legend()

convergence_plot(results_M05_A0)
convergence_plot(results_M05_A125)
