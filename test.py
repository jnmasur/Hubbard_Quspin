import numpy as np
from tools import parameter_instantiate as hhg
from minimization_methods import Parameters, current_expectation_power_spectrum
from matplotlib import pyplot as plt

# Hubbard model Parameters
L = 6  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
pbc = True

# Laser pulse parameters
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm

# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, pbc)
params.set_basis(True)

U = 0 * params.t
a = 1

# contains all important variables
lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=U, t=params.t,
          F0=params.F0, a=a, pbc=params.pbc)

print(lat.field)

# gets times to evaluate at
cycles = 10
n_steps = 2000
start = 0
stop = cycles / lat.freq
times, dt = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)


# def phi(ts):
#     return (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * ts / (2. * cycles)) ** 2.) * np.sin(
#         lat.field * ts)
#
#
# # times = times[:50]
# # plt.plot(times, phi(times), label="phi(t)")
# # plt.plot(times, phi(times + dt), label="phi(t+dt)")
# # plt.show()
#
# freqs, spect1 = current_expectation_power_spectrum((5*params.t, 1), params, True)
# spect2 = current_expectation_power_spectrum((0, 1), params)
# spect3 = current_expectation_power_spectrum((.5*params.t, 1), params)
#
# freqs = freqs / lat.freq
#
# plt.semilogy(freqs, spect1, label="U=5t0")
# plt.semilogy(freqs, spect2, label="U=0t0")
# plt.semilogy(freqs, spect3, label="U=.5t0")
# plt.legend()
# plt.show()
