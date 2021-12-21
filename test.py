import numpy as np
from tools import parameter_instantiate as hhg
from minimization_methods import Parameters, current_expectation_power_spectrum, current_expectation
from evolve import phi_tl
from matplotlib import pyplot as plt

# Hubbard model Parameters
L = 10  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
pbc = True

# Laser pulse parameters
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm

cycles = 10

# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, pbc, cycles)
params.set_basis(True)

U = 7 * params.t
a = 1

# contains all important variables
lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=U, t=params.t,
          F0=params.F0, a=a, pbc=params.pbc)

U1 = 1
U2 = 8
a1 = 1
a2 = 8

freqs, spect1 = current_expectation_power_spectrum((U1 * params.t, a1), params, True)
spect2 = current_expectation_power_spectrum((U2 * params.t, a1), params)
spect3 = current_expectation_power_spectrum((U1 * params.t, a2), params)
spect4 = current_expectation_power_spectrum((U2 * params.t, a2), params)

freqs = freqs / lat.freq

# times, curr1 = current_expectation((U1 * params.t, a1), params, True)
# curr2 = current_expectation((U2 * params.t, a1), params)
# curr3 = current_expectation((U1 * params.t, a2), params)
# curr4 = current_expectation((U2 * params.t, a2), params)

# fig, (ax1, ax2) = plt.subplots(2, 1)

ymin = 1e-13
ymax = 1
plt.semilogy(freqs, spect1 / spect1.max(), label="$U={} \\cdot t_0$, $aF_0 = {} mV$".format(U1, a1 * 100), color="blue")
plt.semilogy(freqs, spect2 / spect2.max(), label="$U={} \\cdot t_0$, $aF_0 = {} mV$".format(U2, a1 * 100), color="green")
plt.semilogy(freqs, spect3 / spect3.max(), label="$U={} \\cdot t_0$, $aF_0 = {} mV$".format(U1, a2 * 100), color="red")
plt.semilogy(freqs, spect4 / spect4.max(), label="$U={} \\cdot t_0$, $aF_0 = {} mV$".format(U2, a2 * 100), color="orange")
# plt.vlines(list(range(1, 12, 2)), ymin, ymax, linestyles="dashed")
plt.ylim((ymin, ymax))
plt.xlim((0, 30))
plt.ylabel("$S(\\omega)$")
plt.xlabel("Harmonic Order")
plt.legend()

# ax2.plot(times, curr1 / curr1.max(), color="blue")
# ax2.plot(times, curr2 / curr2.max(), color="green")
# ax2.plot(times, curr3 / curr3.max(), color="red")
# ax2.plot(times, curr4 / curr4.max(), color="orange")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Current Density")

plt.show()
