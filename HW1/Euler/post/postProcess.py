import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_solution(filename):
    data = pd.read_csv(filename)
    x = data["x"].values
    Q1 = data["Q1"].values
    Q2 = data["Q2"].values
    Q3 = data["Q3"].values
    E1 = data["E1"].values
    E2 = data["E2"].values
    E3 = data["E3"].values
    S1 = data["S1"].values
    S2 = data["S2"].values
    S3 = data["S3"].values
    rho = data["rho"].values
    u = data["u"].values
    p = data["p"].values
    e = data["e"].values
    mach = data["mach"].values
    return x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rho, u, p, e, mach

def read_convergence_history(filename):
    data = pd.read_csv(filename)
    time = data["time"].values
    it = data["it"].values
    res1 = data["res1"].values
    res2 = data["res2"].values
    res3 = data["res3"].values
    return time, it, res1, res2, res3

CFL = ["0-5", "0-75", "1"]
CFL_float = [0.5, 0.75, 1.0]
boundaryCondition = ["1", "2"]


plt.rcParams.update({'font.size': 14})
for bc in boundaryCondition:
    plt.figure()
    for i, cfl in enumerate(CFL):
        filename = f'CFL{cfl}_output{bc}.txt'
        x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rho, u, p, e, mach = read_solution(f'../output/explicit/{filename}')
        plt.plot(x, mach, label=f'CFL={CFL_float[i]}')
    plt.xlabel('Position (x)')
    plt.ylabel('Mach Number')
    plt.title('Mach Number Distribution as function of CFL')
    if bc == "1":
        plt.yticks([1.25, 1.5, 1.75, 2.0])
    if bc == "2":
        plt.yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    plt.legend()
    plt.grid()
    plt.show()

for bc in boundaryCondition:
    plt.figure()
    for i, cfl in enumerate(CFL):
        filename = f'CFL{cfl}_output{bc}.txt'
        time, it, res1, res2, res3 = read_convergence_history(f'../output/explicit/convergence_{filename}')
        plt.semilogy(it, res1, label=f'CFL={CFL_float[i]}')
    plt.xlabel('Iteration')
    plt.ylabel('Residuals Q1')
    plt.title('Convergence History as function of CFL')
    plt.legend()
    plt.grid()
    plt.show()


# plt.figure()
# plt.plot(x, rho, label='Density (rho)')
# plt.xlabel('Position (x)')
# plt.ylabel('Density (rho)')
# plt.title('Density Distribution')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(x, u, label='Velocity (u)', color='orange')
# plt.xlabel('Position (x)')
# plt.ylabel('Velocity (u)')
# plt.title('Velocity Distribution')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(x, p, label='Pressure (p)', color='green')
# plt.xlabel('Position (x)')
# plt.ylabel('Pressure (p)')
# plt.title('Pressure Distribution')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(x, e, label='Energy (e)', color='red')
# plt.xlabel('Position (x)')
# plt.ylabel('Energy (e)')
# plt.title('Energy Distribution')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(x, mach, label='Mach Number', color='purple')
# plt.xlabel('Position (x)')
# plt.ylabel('Mach Number')
# plt.title('Mach Number Distribution')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure()
# plt.semilogy(it, res1, label='Residual Q1', color='blue')
# plt.semilogy(it, res2, label='Residual Q2', color='orange')
# plt.semilogy(it, res3, label='Residual Q3', color='green')
# plt.xlabel('Iteration')
# plt.ylabel('Residuals')
# plt.title('Convergence History')
# plt.legend()
# plt.grid()
# plt.show()

CFL = [1, 3, 5, 10, 15, 20, 25]
plt.figure()
for cfl in CFL:
    filename = f'implicit_CFL{cfl}_output1.txt'
    x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rho, u, p, e, mach = read_solution(f'../output/implicit/{filename}')
    plt.plot(x, mach, label=f'CFL={cfl}')
plt.xlabel('Position (x)')
plt.ylabel('Mach Number')
plt.title(f'Mach Number Distribution for Implicit Scheme')
plt.legend()
plt.grid()
plt.show()

plt.figure()
for cfl in CFL:
    filename = f'implicit_CFL{cfl}_output1.txt'
    time, it, res1, res2, res3 = read_convergence_history(f'../output/implicit/convergence_{filename}')

    plt.loglog(it, res1, label=f'CFL={cfl}')
plt.xlabel('Iteration')
plt.ylabel('Residuals Q1')
plt.title(f'Convergence History for Implicit Scheme')
plt.legend()
plt.grid()
plt.show()

filename = f'implicit_CFL1_e1_output2.txt'
x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rho, u, p, e, mach = read_solution(f'../{filename}')
plt.figure()
plt.plot(x, mach)
plt.xlabel('Position (x)')
plt.ylabel('Mach Number')
plt.title(f'Mach Number Distribution')
plt.yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
plt.grid()
plt.show()