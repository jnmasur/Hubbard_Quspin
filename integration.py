from evolve import phi_tl
from tools import parameter_instantiate as hhg
import numpy as np
from scipy.integrate import dblquad, quad
from matplotlib import pyplot as plt
from time import time

L = 6  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
pbc = True
U = 1 * t0

# Laser pulse parameters
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm

# at this point, lat is only useful for finding the frequency which has no dependence on a
lat = hhg(field, N_up, N_down, L, 0, U, t=t0, F0=F0, a=0)
cycles = 10
n_steps = 2000
start = 0
stop = cycles / lat.freq
stop += 0

a_values = np.linspace(1, 10, num=100)
omega_1_real_values = []
err_1_real_values = []
omega_1_imag_values = []
err_1_imag_values = []
omega_2_real_values = []
err_2_real_values = []
omega_2_imag_values = []
err_2_imag_values = []

ti = time()
for a in a_values:
    print(a)
    lat = hhg(field, N_up, N_down, L, 0, U, t=t0, F0=F0, a=a)

    phi = lambda t: phi_tl(t, lat, cycles)

    # split the functions into real and imaginary such that: int(f(x)) = int(Re[f(x)]) + i*int(Im[f(x)])
    omega_1_func_real = lambda t: np.cos(phi(t))
    omega_1_func_imag = lambda t: -1 * np.sin(phi(t))
    omega_2_func_real = lambda t2, t1: np.cos(phi(t1)) - np.cos(phi(t2))
    omega_2_func_imag = lambda t2, t1: np.sin(phi(t2)) - np.sin(phi(t1))

    # noinspection PyTupleAssignmentBalance
    omega_1_real, err_1_real = quad(omega_1_func_real, start, stop)
    # noinspection PyTupleAssignmentBalance
    omega_1_imag, err_1_imag = quad(omega_1_func_imag, start, stop)
    omega_2_real, err_2_real = dblquad(omega_2_func_real, start, stop, lambda x: 0, lambda x: x)
    omega_2_imag, err_2_imag = dblquad(omega_2_func_imag, start, stop, lambda x: 0, lambda x: x)

    omega_1_real_values.append(omega_1_real)
    err_1_real_values.append(err_1_real)
    omega_1_imag_values.append(omega_1_imag)
    err_1_imag_values.append(err_1_imag)
    omega_2_real_values.append(omega_2_real)
    err_2_real_values.append(err_2_real)
    omega_2_imag_values.append(omega_2_imag)
    err_2_imag_values.append(err_2_imag)

print("Total time =", time() - ti)

np.save("./IntegrationData/Omega1RealValues.npy", np.array(omega_1_real_values))
np.save("./IntegrationData/Omega1ImaginaryValues.npy", np.array(omega_1_imag_values))
np.save("./IntegrationData/Omega2RealValues.npy", np.array(omega_2_real_values))
np.save("./IntegrationData/Omega2ImagValues.npy", np.array(omega_2_imag_values))
np.save("./IntegrationData/ErrorInOmega1Real.npy", np.array(err_1_real_values))
np.save("./IntegrationData/ErrorInOmega1Imaginary.npy", np.array(err_1_imag_values))
np.save("./IntegrationData/ErrorInOmega2Real.npy", np.array(err_2_real_values))
np.save("./IntegrationData/ErrorInOmega2Imaginary.npy", np.array(err_2_imag_values))

omega_1_real_values = np.load("./IntegrationData/Omega1RealValues.npy")
omega_1_imag_values = np.load("./IntegrationData/Omega1ImaginaryValues.npy")
omega_2_real_values = np.load("./IntegrationData/Omega2RealValues.npy")
omega_2_imag_values = np.load("./IntegrationData/Omega2ImagValues.npy")
err_1_real_values = np.load("./IntegrationData/ErrorInOmega1Real.npy")
err_1_imag_values = np.load("./IntegrationData/ErrorInOmega1Imaginary.npy")
err_2_real_values = np.load("./IntegrationData/ErrorInOmega2Real.npy")
err_2_imag_values = np.load("./IntegrationData/ErrorInOmega2Imaginary.npy")

plt.errorbar(a_values, omega_1_real_values, yerr=err_1_real_values, label="$Re[\\Omega_1(T)]$")
plt.errorbar(a_values, omega_1_imag_values, yerr=err_1_imag_values, label="$Im[\\Omega_1(T)]$")
plt.errorbar(a_values, omega_2_real_values, yerr=err_2_real_values, label="$Re[\\Omega_2(T)$]")
plt.errorbar(a_values, omega_2_imag_values, yerr=err_2_imag_values, label="$Im[\\Omega_2(T)$]")
plt.title("First two terms in ME (0 to $\\frac{2\\pi N}{\\omega_0}$)")
plt.xlabel("a values")
plt.ylabel("Integrals")
plt.legend()
plt.show()
