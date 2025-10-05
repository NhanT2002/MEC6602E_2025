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
    # plt.title('Mach Number Distribution as function of CFL')
    if bc == "1":
        plt.yticks([1.25, 1.5, 1.75, 2.0])
    if bc == "2":
        plt.yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'mach_bc{bc}.pdf', format='pdf')

for bc in boundaryCondition:
    plt.figure()
    for i, cfl in enumerate(CFL):
        filename = f'CFL{cfl}_output{bc}.txt'
        time, it, res1, res2, res3 = read_convergence_history(f'../output/explicit/convergence_{filename}')
        plt.semilogy(it, res1, label=f'CFL={CFL_float[i]}')
    plt.xlabel('Iteration')
    plt.ylabel('Residuals Q1')
    # plt.title('Convergence History as function of CFL')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'convergence_bc{bc}.pdf', format='pdf')

CFL = [1, 3, 5, 10, 15, 20, 25]
plt.figure()
for cfl in CFL:
    filename = f'implicit_CFL{cfl}_output1.txt'
    x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rho, u, p, e, mach = read_solution(f'../output/implicit/{filename}')
    plt.plot(x, mach, label=f'CFL={cfl}')
plt.xlabel('Position (x)')
plt.ylabel('Mach Number')
plt.yticks([1.25, 1.5, 1.75, 2.0])
# plt.title(f'Mach Number Distribution for Implicit Scheme')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('mach_implicit_bc1.pdf', format='pdf')

plt.figure()
for cfl in CFL:
    filename = f'implicit_CFL{cfl}_output1.txt'
    time, it, res1, res2, res3 = read_convergence_history(f'../output/implicit/convergence_{filename}')

    plt.loglog(it, res1, label=f'CFL={cfl}')
plt.xlabel('Iteration')
plt.ylabel('Residuals Q1')
# plt.title(f'Convergence History for Implicit Scheme')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('convergence_implicit_bc1.pdf', format='pdf')

plt.figure()
for cfl in [1, 3, 5, 10]:
    filename = f'implicit_CFL{cfl}_e0-1_output2.txt'
    x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rho, u, p, e, mach = read_solution(f'../output/implicit/{filename}')
    plt.plot(x, mach, label=f'CFL={cfl}, ε=0.1')
filename = f'implicit_CFL1_e1_output2.txt'
x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rho, u, p, e, mach = read_solution(f'../output/implicit/{filename}')
plt.plot(x, mach, label=f'CFL={1}, ε=1.0')
plt.xlabel('Position (x)')
plt.ylabel('Mach Number')
plt.yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
# plt.title(f'Mach Number Distribution for Implicit Scheme with Different Epsilon')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('mach_implicit_bc2.pdf', format='pdf')

plt.figure()
for cfl in [1, 3, 5, 10]:
    filename = f'implicit_CFL{cfl}_e0-1_output2.txt'
    time, it, res1, res2, res3 = read_convergence_history(f'../output/implicit/convergence_{filename}')
    plt.loglog(it, res1, label=f'CFL={cfl}, ε=0.1')
filename = f'implicit_CFL1_e1_output2.txt'
time, it, res1, res2, res3 = read_convergence_history(f'../output/implicit/convergence_{filename}')
plt.loglog(it, res1, label=f'CFL={1}, ε=1.0')
plt.xlabel('Iteration')
plt.ylabel('Residuals Q1')
# plt.title(f'Convergence History for Implicit Scheme with Different Epsilon')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('convergence_implicit_bc2.pdf', format='pdf')

def A(x) :
    return 1.398 + 0.347 * np.tanh(0.8 * x - 4)

x = np.linspace(0, 10, 1000)
plt.figure(figsize=(8,3))
plt.plot(x, A(x))
plt.xlabel('Position (x)')
plt.ylabel('A(x)')
# plt.title('Nozzle Area Distribution')
plt.grid()
plt.tight_layout()
plt.savefig('nozzle_area.pdf', format='pdf')


filename_explicit_CFL1_bc1 = '../output/comparaison/convergence_explicit_CFL1_output1.txt'
filename_explicit_CFL1_bc2 = '../output/comparaison/convergence_explicit_CFL1_output2.txt'
filename_implicit_CFL25_bc1 = '../output/comparaison/convergence_implicit_CFL25_output1.txt'
filename_implicit_CFL10_bc2 = '../output/comparaison/convergence_implicit_CFL10_output2.txt'
files = [filename_explicit_CFL1_bc1, filename_implicit_CFL25_bc1, filename_explicit_CFL1_bc2, filename_implicit_CFL10_bc2]
labels = ['Explicit CFL=1, BC=1', 'Implicit CFL=25, BC=1', 'Explicit CFL=1, BC=2', 'Implicit CFL=10, BC=2']

plt.figure()
for filename, label in zip(files, labels):
    time, it, res1, res2, res3 = read_convergence_history(f'{filename}')
    plt.loglog(time, res1, label=label)
plt.xlabel('Time (s)')
plt.ylabel('Residuals Q1')
# plt.title('Convergence History Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('convergence_comparison.pdf', format='pdf')

plt.figure()
for filename, label in zip(files, labels):
    time, it, res1, res2, res3 = read_convergence_history(f'{filename}')
    plt.loglog(it, res1, label=label)
plt.xlabel('Iteration')
plt.ylabel('Residuals Q1')
# plt.title('Convergence History Comparison')
plt.legend(loc='center left')
plt.grid()
plt.tight_layout()
plt.savefig('convergence_comparison_it.pdf', format='pdf')