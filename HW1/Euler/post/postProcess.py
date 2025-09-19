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
    it = data["it"].values
    res1 = data["res1"].values
    res2 = data["res2"].values
    res3 = data["res3"].values
    return it, res1, res2, res3

x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rho, u, p, e, mach = read_solution('../output.txt')
it, res1, res2, res3 = read_convergence_history('../convergence_output.txt')

plt.figure()
plt.plot(x, rho, label='Density (rho)')
plt.xlabel('Position (x)')
plt.ylabel('Density (rho)')
plt.title('Density Distribution')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(x, u, label='Velocity (u)', color='orange')
plt.xlabel('Position (x)')
plt.ylabel('Velocity (u)')
plt.title('Velocity Distribution')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(x, p, label='Pressure (p)', color='green')
plt.xlabel('Position (x)')
plt.ylabel('Pressure (p)')
plt.title('Pressure Distribution')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(x, e, label='Energy (e)', color='red')
plt.xlabel('Position (x)')
plt.ylabel('Energy (e)')
plt.title('Energy Distribution')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(x, mach, label='Mach Number', color='purple')
plt.xlabel('Position (x)')
plt.ylabel('Mach Number')
plt.title('Mach Number Distribution')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.semilogy(it, res1, label='Residual Q1', color='blue')
plt.semilogy(it, res2, label='Residual Q2', color='orange')
plt.semilogy(it, res3, label='Residual Q3', color='green')
plt.xlabel('Iteration')
plt.ylabel('Residuals')
plt.title('Convergence History')
plt.legend()
plt.grid()
plt.show()