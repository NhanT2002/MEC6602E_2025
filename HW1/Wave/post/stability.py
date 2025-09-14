import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 1000)

# Explicit Backward Euler method for the 1D wave equation
def G_explicit_backward(sigma, x) :
    g_complex = 1 - sigma * (1 - np.exp(-1j*x))
    return g_complex.real**2 + g_complex.imag**2

plt.figure()
sigma_values = [0.5, 0.75, 1.0, 1.01]
for sigma in sigma_values:
    G_values = G_explicit_backward(sigma, x)
    plt.plot(x, G_values, label=f'σ = {sigma}')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.title("Stability Analysis of Explicit Backward Euler Method")
plt.grid()
plt.legend()

# Explicit Forward Euler method for the 1D wave equation
def G_explicit_forward(sigma, x) :
    g_complex = 1 - sigma * (np.exp(1j*x)-1)
    return g_complex.real**2 + g_complex.imag**2

plt.figure()
sigma_values = [-0.5, -1.0, 0.5, 1.0]
for sigma in sigma_values:
    G_values = G_explicit_forward(sigma, x)
    plt.plot(x, G_values, label=f'σ = {sigma}')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.title("Stability Analysis of Explicit Forward Euler Method")
plt.grid()
plt.legend()

# Forward Euler - Centered Space method for the 1D wave equation
def G_forward_centered(sigma, x) :
    g_complex = 1 - sigma/2 * (np.exp(1j*x)-np.exp(-1j*x))
    return g_complex.real**2 + g_complex.imag**2

plt.figure()
sigma_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
for sigma in sigma_values:
    G_values = G_forward_centered(sigma, x)
    plt.plot(x, G_values, label=f'σ = {sigma}', linestyle='--' if sigma > 0 else '-')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.title("Stability Analysis of Forward Euler - Centered Space Method")
plt.grid()
plt.legend()

# Leapfrog - Centered Space method for the 1D wave equation
def G_leapfrog_centered(sigma, x) :
    g_complex_plus = -sigma*1j*np.sin(x) + np.emath.sqrt(1 - sigma**2 * np.sin(x)**2)
    g_complex_minus = -sigma*1j*np.sin(x) - np.emath.sqrt(1 - sigma**2 * np.sin(x)**2)
    return g_complex_plus.real**2 + g_complex_plus.imag**2, g_complex_minus.real**2 + g_complex_minus.imag**2

plt.figure()
sigma_values = [0.0, 0.5, 1.0, 1.1]
for sigma in sigma_values:
    G_plus_values, G_minus_values = G_leapfrog_centered(sigma, x)
    plt.plot(x, G_plus_values, label=f'σ = {sigma} (plus)')
    plt.plot(x, G_minus_values, label=f'σ = {sigma} (minus)')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.title("Stability Analysis of Leapfrog - Centered Space Method")
plt.grid()
plt.legend()