import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import read_PLOT3D_mesh, read_plot3d_2d, compute_coeff, read_residual_history, compute_order

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
        plt.plot(x[0], cp_airfoil, '--', label="Euler Solver 2048x2048", color='tab:orange')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Cp")
        plt.title(f"NACA0012 Mach={m} alpha={a}°")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"fig/NACA0012_M{MACH[m]}_A{ALPHA[a]}_Cp_comparison.pdf")

def read_residual_history(file_name):
    data = pd.read_csv(file_name, sep=",")
    Time = data['Time'].values
    R0 = data['Residual_0'].values
    R1 = data['Residual_1'].values
    R2 = data['Residual_2'].values
    R3 = data['Residual_3'].values
    cl = data['cl'].values
    cd = data['cd'].values
    cm = data['cm'].values
    return Time, R0, R1, R2, R3, cl, cd, cm

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

# Residual history plotting
for a in ALPHA.keys() : 
    for m in MACH.keys() :
        plt.figure()
        for size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048] :
            Time, R0, R1, R2, R3, cl, cd, cm = read_residual_history(f"../residual_history/residual_history_M{MACH[m]}_A{ALPHA[a]}_{size}.txt")
            plt.loglog(Time, R0, label=f"{size}x{size}")
        plt.xlabel("Time (s)")
        plt.ylabel("Residual continuity")
        plt.title(f"Residual history NACA0012 Mach={m} alpha={a}°")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fig/residual_history_NACA0012_M{MACH[m]}_A{ALPHA[a]}_loglog.pdf")


# Read results
results_M05_A0 = pd.read_csv("results_M05_A0.csv")
results_M05_A125 = pd.read_csv("results_M05_A125.csv")
results_M08_A0 = pd.read_csv("results_M08_A0.csv")
results_M08_A125 = pd.read_csv("results_M08_A125.csv")
results_M08_A125_Jameson = pd.read_csv("results_M08_A125_Jameson.csv")

plt.rcParams.update({'font.size': 12})
def convergence_plot(results, coef : str ='C_D') :
    plt.figure()
    
    # compute the order of convergence and plot it
    cd = np.array(results[coef])[-3:]
    _, cd_star = compute_order(cd[-1], cd[-2], cd[-3])

    # cd_star = cd[-1] + (cd[-1] - cd[-2]) / (2**2 - 1)
    # errors = np.abs(results[coef][-2:] - cd_star)
    # hs = 1 / results['Mesh Size'][-2:]

    # plot the fit line with the last value matched
    errors = np.abs(results[coef][-3:] - cd_star)
    hs = 1 / results['Mesh Size'][-3:]

    p = np.polyfit(np.log(hs), np.log(errors), 1)

    plt.loglog(1/results['Mesh Size'], np.abs(results[coef]- cd_star), '-o')
    # add mesh size next to each point
    for i, size in enumerate(results['Mesh Size']) :
        plt.text(1/size, np.abs(results[coef]- cd_star)[i], f"{size}", fontsize=12, ha='right', va='bottom')
    plt.loglog(1/results['Mesh Size'], np.exp(p[1]) * (1/results['Mesh Size'])**p[0], '--', label=f"Order = {p[0]:.3f}")
    
    plt.legend()
    plt.xlabel("1 / NC")
    plt.ylabel(f"$|{coef} - {coef}^*|$")
    plt.xlim(1/4000, 1/5)
    plt.grid()
    plt.title(f"${coef}^*$ = {cd_star:.9f}")

convergence_plot(results_M05_A0, coef='C_d')
plt.savefig("fig/convergence_M05_A0_C_D.pdf")

convergence_plot(results_M05_A125, coef='C_l')
plt.savefig("fig/convergence_M05_A125_C_L.pdf")
convergence_plot(results_M05_A125, coef='C_d')
plt.savefig("fig/convergence_M05_A125_C_D.pdf")
convergence_plot(results_M05_A125, coef='C_m')
plt.savefig("fig/convergence_M05_A125_C_M.pdf")

convergence_plot(results_M08_A0, coef='C_d')
plt.savefig("fig/convergence_M08_A0_C_D.pdf")

convergence_plot(results_M08_A125, coef='C_l')
plt.savefig("fig/convergence_M08_A125_C_L.pdf")
convergence_plot(results_M08_A125, coef='C_d')
plt.savefig("fig/convergence_M08_A125_C_D.pdf")
convergence_plot(results_M08_A125, coef='C_m')
plt.savefig("fig/convergence_M08_A125_C_M.pdf")


# Multigrid plot
x, y = read_PLOT3D_mesh("../mesh/naca0012_512x512.xyz")
Time_mg, R0_mg, R1_mg, R2_mg, R3_mg, cl_mg, cd_mg, cm_mg = read_residual_history(f"../output/residual_history_M12_A0_256_multigrid.txt")
Time_rs, R0_rs, R1_rs, R2_rs, R3_rs, cl_rs, cd_rs, cm_rs = read_residual_history(f"../output/residual_history_M12_A0_256_res_smooth.txt")

plt.figure()
plt.semilogy(R0_mg, label="With Multigrid")
plt.semilogy(R0_rs, label="Without Multigrid")
plt.xlabel("Iterations")
plt.ylabel("Residual continuity")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("fig/residual_multigrid_comparison.pdf")

plt.figure()
plt.semilogy(Time_mg, R0_mg, label="With Multigrid")
plt.semilogy(Time_rs, R0_rs, label="Without Multigrid")
plt.xlabel("Time (s)")
plt.ylabel("Residual continuity")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("fig/residual_multigrid_comparison_time.pdf")