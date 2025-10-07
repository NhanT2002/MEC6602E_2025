import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
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
plt.title("Explicit Backward Euler Method")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("stability_explicit_backward_euler.pdf", format='pdf')

# Explicit Forward Euler method for the 1D wave equation
def G_explicit_forward(sigma, x) :
    g_complex = 1 - sigma * (np.exp(1j*x)-1)
    return g_complex.real**2 + g_complex.imag**2

plt.figure()
sigma_values = [0.25, 0.5, 1.0]
for sigma in sigma_values:
    G_values = G_explicit_forward(sigma, x)
    plt.plot(x, G_values, label=f'σ = {sigma}')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.title("Explicit Forward Euler Method")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("stability_explicit_forward_euler.pdf", format='pdf')

# Forward Euler - Centered Space method for the 1D wave equation
def G_forward_centered(sigma, x) :
    g_complex = 1 - sigma/2 * (np.exp(1j*x)-np.exp(-1j*x))
    return g_complex.real**2 + g_complex.imag**2

plt.figure()
sigma_values = [0.25, 0.5, 1.0]
for sigma in sigma_values:
    G_values = G_forward_centered(sigma, x)
    plt.plot(x, G_values, label=f'σ = {sigma}')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.title("Forward Euler - Centered Space Method")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("stability_forward_centered.pdf", format='pdf')

# Leapfrog - Centered Space method for the 1D wave equation
def G_leapfrog_centered(sigma, x) :
    g_complex_plus = -sigma*1j*np.sin(x) + np.emath.sqrt(1 - sigma**2 * np.sin(x)**2)
    g_complex_minus = -sigma*1j*np.sin(x) - np.emath.sqrt(1 - sigma**2 * np.sin(x)**2)
    return g_complex_plus.real**2 + g_complex_plus.imag**2, g_complex_minus.real**2 + g_complex_minus.imag**2

plt.figure()
sigma_values = [0.25, 0.5, 1.0, 1.01]
for sigma in sigma_values:
    G_plus_values, G_minus_values = G_leapfrog_centered(sigma, x)
    plt.plot(x, G_plus_values, label=f'σ = {sigma} (plus)')
    plt.plot(x, G_minus_values, label=f'σ = {sigma} (minus)')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.title("Leapfrog - Centered Space Method")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("stability_leapfrog_centered.pdf", format='pdf')

# Lax-Wendroff method for the 1D wave equation
def G_lax_wendroff(sigma, x) :
    g_complex = 1 - sigma/2*(2j*np.sin(x)) + (sigma**2)/2 * (-2 + 2*np.cos(x))
    return g_complex.real**2 + g_complex.imag**2
plt.figure()
sigma_values = [0.25, 0.5, 1.0, 1.1]
for sigma in sigma_values:
    G_values = G_lax_wendroff(sigma, x)
    plt.plot(x, G_values, label=f'σ = {sigma}')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.title("Lax-Wendroff Method")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("stability_lax_wendroff.pdf", format='pdf')

# Lax method for the 1D wave equation
def G_lax(sigma, x) :
    g_complex = np.cos(x) - 1j*sigma*np.sin(x)
    return g_complex.real**2 + g_complex.imag**2
plt.figure()
sigma_values = [0.25, 0.5, 1.0, 1.1]
for sigma in sigma_values:
    G_values = G_lax(sigma, x)
    plt.plot(x, G_values, label=f'σ = {sigma}')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.title("Lax Method")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("stability_lax.pdf", format='pdf')

# Hybrid explicit-implicit method for the 1D wave equation
def G_hybrid(sigma, x, theta) :
    g_complex = (1 - (1-theta)*sigma*1j*np.sin(x)) / (1 + theta*sigma*1j*np.sin(x))
    return g_complex.real**2 + g_complex.imag**2

theta_values = [0.0, 0.5, 1.0]
sigma_values = [0.5, 1.0, 1.5, 2.0]
for theta in theta_values:
    plt.figure()
    for sigma in sigma_values:
        G_values = G_hybrid(sigma, x, theta)
        plt.plot(x, G_values, label=f'σ = {sigma}')
    plt.xlabel(r"$k_m\Delta x$")
    plt.ylabel(r"$|G|^2$")
    plt.title(f"Hybrid Method (θ = {theta})")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"stability_hybrid_theta_{theta}.pdf", format='pdf')

# 2nd order space, 4th order time method for the 1D wave equation
def G_2nd4th(sigma, x) :
    # g_complex = 1 - sigma/2*(np.exp(1j*x) - np.exp(-1j*x)) + (sigma**2)/2*(-2 + np.exp(1j*x) + np.exp(-1j*x)) - (sigma**3)/6 * (-0.5*np.exp(-2j*x) + np.exp(-1j*x) - np.exp(1j*x) + 0.5*np.exp(2j*x)) + (sigma**4)/24 * (6 - 4*np.exp(-1j*x) + np.exp(-2j*x) + np.exp(2j*x) - 4*np.exp(1j*x))
    g_complex = 1 - sigma*1j*np.sin(x) + sigma**2 * (np.cos(x) - 1) - sigma**3/6 * (1j * np.sin(2*x) -2j*np.sin(x)) + sigma**4/24 * (6 + 2*np.cos(2*x) - 8*np.cos(x))
    return g_complex.real**2 + g_complex.imag**2
plt.figure()
sigma_values = [0.5, 1.0, 1.5, 1.7, 1.75]
for sigma in sigma_values:
    G_values = G_2nd4th(sigma, x)
    plt.plot(x, G_values, label=f'σ = {sigma}')
plt.xlabel(r"$k_m\Delta x$")
plt.ylabel(r"$|G|^2$")
plt.title("2nd Order Space, 4th Order Time Method")
plt.grid()
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("stability_2nd4th.pdf", format='pdf')