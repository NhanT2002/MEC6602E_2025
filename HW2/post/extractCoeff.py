import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import read_PLOT3D_mesh, read_plot3d_2d, compute_coeff, read_residual_history

mesh_size = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Create a table to store CL, CD, CM for different mesh sizes and conditions
df_results_M05_A0 = []
for size in mesh_size :
    x, y = read_PLOT3D_mesh(f"../mesh/naca0012_{size}x{size}.xyz")
    ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(f"../output/output_M05_A0_{size}.q")
    cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    df_results_M05_A0.append({'Mesh Size': size, 'C_l': C_L, 'C_d': C_D, 'C_m': C_M})
df_results_M05_A0 = pd.DataFrame(df_results_M05_A0)
df_results_M05_A0.to_csv("results_M05_A0.csv", index=False)

df_results_M05_A125 = []
for size in mesh_size :
    x, y = read_PLOT3D_mesh(f"../mesh/naca0012_{size}x{size}.xyz")
    ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(f"../output/output_M05_A125_{size}.q")
    cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    df_results_M05_A125.append({'Mesh Size': size, 'C_l': C_L, 'C_d': C_D, 'C_m': C_M})
df_results_M05_A125 = pd.DataFrame(df_results_M05_A125)
df_results_M05_A125.to_csv("results_M05_A125.csv", index=False)

df_results_M08_A0 = []
for size in mesh_size :
    x, y = read_PLOT3D_mesh(f"../mesh/naca0012_{size}x{size}.xyz")
    ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(f"../output/output_M08_A0_{size}.q")
    cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    df_results_M08_A0.append({'Mesh Size': size, 'C_l': C_L, 'C_d': C_D, 'C_m': C_M})
df_results_M08_A0 = pd.DataFrame(df_results_M08_A0)
df_results_M08_A0.to_csv("results_M08_A0.csv", index=False)

df_results_M08_A125 = []
for size in mesh_size :
    x, y = read_PLOT3D_mesh(f"../mesh/naca0012_{size}x{size}.xyz")
    ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(f"../output/output_M08_A125_{size}.q")
    cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    df_results_M08_A125.append({'Mesh Size': size, 'C_l': C_L, 'C_d': C_D, 'C_m': C_M})
df_results_M08_A125 = pd.DataFrame(df_results_M08_A125)
df_results_M08_A125.to_csv("results_M08_A125.csv", index=False)