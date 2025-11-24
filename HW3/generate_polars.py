import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from helper import read_PLOT3D_mesh, read_plot3d_2d, compute_coeff, compute_order



def read_residual_history(file_name):
    data = pd.read_csv(file_name, sep=",", skiprows=1, header=None)
    data.columns = ['Time', 'Residual_0', 'Residual_1', 'Residual_2', 'Residual_3', 'cl', 'cd', 'cm', " "]
    Time = data['Time'].values
    R0 = data['Residual_0'].values
    R1 = data['Residual_1'].values
    R2 = data['Residual_2'].values
    R3 = data['Residual_3'].values
    cl = data['cl'].values
    cd = data['cd'].values
    cm = data['cm'].values
    return Time, R0, R1, R2, R3, cl, cd, cm

# increase font size for plots
plt.rcParams.update({'font.size': 14})


MESHES = ["naca2412", "sc20712", "crm"]
MACH = [0.2, 0.5, 0.7]
ANGLES = np.arange(-5.0, 16.0, 0.5)

for mesh in MESHES :
    fig1, ax1 = plt.subplots() # cl plot
    fig2, ax2 = plt.subplots() # cd plot
    fig3, ax3 = plt.subplots() # cm plot
    for mach in MACH :
        mach_str = f"{mach:.3f}"
        CL = []
        CD = []
        CM = []
        x, y = read_PLOT3D_mesh(f"mesh/{mesh}.xyz")

        for a in ANGLES :
            output_file = f"solver_outputs/{mesh}/mach_{mach_str}/aoa_{a:.3f}/output_aoa_{a:.3f}.q"
            # check if output file exists
            if not os.path.exists(output_file):
                print(f"Warning: Output file not found: {output_file}. Skipping this case.")
                CL.append(np.nan)
                CD.append(np.nan)
                CM.append(np.nan)
                continue
            else :
                ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(output_file)
                Time, R0, R1, R2, R3, cl_res, cd_res, cm_res = read_residual_history(f"solver_outputs/{mesh}/mach_{mach_str}/aoa_{a:.3f}/residual_history_aoa_{a:.3f}.txt")
                # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
            C_L = cl_res[-1]
            C_D = cd_res[-1]
            C_M = cm_res[-1]
            CL.append(C_L)
            CD.append(C_D)
            CM.append(C_M)

        # save polar data to CSV
        df = pd.DataFrame({
            'aoa': ANGLES,
            'cl': CL,
            'cd': CD,
            'cm': CM
        })
        df.to_csv(f"polars/{mesh}_polar_mach_{mach_str}.csv", index=False)

        ax1.plot(ANGLES, CL, '-o', label=f'Mach {mach_str}')
        ax2.plot(ANGLES, CD, '-o', label=f'Mach {mach_str}')
        ax3.plot(ANGLES, CM, '-o', label=f'Mach {mach_str}')

    ax1.set_xlabel("$\\alpha$ (deg)")
    ax1.set_ylabel("$C_l$")
    ax1.set_title("Lift Coefficient vs Angle of Attack")
    ax1.grid()
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"polars/{mesh}_lift_coefficient.pdf")

    ax2.set_xlabel("$\\alpha$ (deg)")
    ax2.set_ylabel("$C_d$")
    ax2.set_title("Drag Coefficient vs Angle of Attack")
    ax2.grid()
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"polars/{mesh}_drag_coefficient.pdf")

    ax3.set_xlabel("$\\alpha$ (deg)")
    ax3.set_ylabel("$C_m$")
    ax3.set_title("Moment Coefficient vs Angle of Attack")
    ax3.grid()
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(f"polars/{mesh}_moment_coefficient.pdf")

# Mach polars at iso-cl
for mesh in MESHES :
    C_L_iso_cl = []
    C_D_iso_cl = []
    C_M_iso_cl = []
    alpha_iso_cl = []
    mach_values = []
    
    for mach in np.arange(0.2, 0.8, 0.05) :
        mach_str = f"{mach:.3f}"
        # list files in /home/apollon/hitra2/MEC6602E/HW3/solver_outputs_iso_cl/crm/mach_0.200
        file_list = os.listdir(f"solver_outputs_iso_cl/{mesh}/mach_{mach_str}")
        # target the file that starts with residual_history_aoa_
        for file_name in file_list :
            if file_name.startswith("residual_history_aoa_") :
                res_file = f"solver_outputs_iso_cl/{mesh}/mach_{mach_str}/{file_name}"
                mach_, alpha_ = res_file.split(".txt")[0].split("_")[-2:]
                mach = float(mach_.split("-")[-1])
                alpha = float(alpha_.split("-")[-1])
                Time, R0, R1, R2, R3, cl_res, cd_res, cm_res = read_residual_history(res_file)
                C_L_iso_cl.append(cl_res[-1])
                C_D_iso_cl.append(cd_res[-1])
                C_M_iso_cl.append(cm_res[-1])
                alpha_iso_cl.append(alpha)
                mach_values.append(mach)

    df_iso_cl = pd.DataFrame({
        'mach': mach_values,
        'alpha': alpha_iso_cl,
        'cl': C_L_iso_cl,
        'cd': C_D_iso_cl,
        'cm': C_M_iso_cl
    })
    df_iso_cl.to_csv(f"polars/{mesh}_polar_iso_cl.csv", index=False)
    
    plt.figure()
    plt.plot(mach_values, C_L_iso_cl, '-o')
    plt.xlabel("Mach Number")
    plt.ylabel("$C_l$ at iso-$C_l$")
    plt.title(f"Lift Coefficient at iso-$C_l$ vs Mach Number for {mesh}")
    plt.grid()
    plt.savefig(f"polars/{mesh}_lift_coefficient_iso_cl.pdf")

    plt.figure()
    plt.plot(mach_values, C_D_iso_cl, '-o')
    plt.xlabel("Mach Number")
    plt.ylabel("$C_d$ at iso-$C_l$")
    plt.title(f"Drag Coefficient at iso-$C_l$ vs Mach Number for {mesh}")
    plt.grid()
    plt.savefig(f"polars/{mesh}_drag_coefficient_iso_cl.pdf")

    plt.figure()
    plt.plot(mach_values, C_M_iso_cl, '-o')
    plt.xlabel("Mach Number")
    plt.ylabel("$C_m$ at iso-$C_l$")
    plt.title(f"Moment Coefficient at iso-$C_l$ vs Mach Number for {mesh}")
    plt.grid()
    plt.savefig(f"polars/{mesh}_moment_coefficient_iso_cl.pdf")

        