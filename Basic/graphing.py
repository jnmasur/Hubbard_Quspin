from matplotlib import pyplot as plt
import numpy as np
from tools import parameter_instantiate as hhg

"""Hubbard model Parameters"""
L = 6 # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
U = 1 * t0  # interaction strength
pbc = True

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm
a = 4  # Lattice constant Angstroms

"""Parameters for approximating ground state"""
delta_U = 0.0001
delta_a = 0.001

lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a, pbc=pbc)

lat_plus_dU = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U+(delta_U/2), t=t0, F0=F0, a=a, pbc=pbc)
lat_minus_dU = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U-(delta_U/2), t=t0, F0=F0, a=a, pbc=pbc)

lat_plus_da = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a+(delta_a/2), pbc=pbc)
lat_minus_da = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a-(delta_a/2), pbc=pbc)

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
stop = cycles / lat.freq
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

"""set up parameters for saving expectations later"""
parameters = '-{}sites-{}t0-{}U-{}delta_U-{}a-{}delta_a-{}field-{}amplitude-{}cycles-{}steps-{}pbc'.format(L,t0,U,delta_U,a,delta_a,field,F0,cycles,n_steps,pbc)

delta_U = lat_plus_dU.U - lat_minus_dU.U
delta_a = lat_plus_da.a - lat_minus_da.a

"""Here we load all expectations"""
H = np.load('./Data/H_expec'+parameters+'.npy')

dH_dU = np.load('./Data/dH_dU_expec'+parameters+'.npy')
H_delta_U = delta_U * dH_dU
H_U_cfd = np.load('./Data/H_U_cfd'+parameters+'.npy')

dH_da = np.load('./Data/dH_da_expec'+parameters+'.npy')
H_delta_a = delta_a * dH_da
H_a_cfd = np.load('./Data/H_a_cfd'+parameters+'.npy')

J = np.load('./Data/J_expec'+parameters+'.npy')

dJ_dU = np.load('./Data/dJ_dU_expec'+parameters+'.npy')
J_delta_U = delta_U * dJ_dU
J_U_cfd = np.load('./Data/J_U_cfd'+parameters+'.npy')

dJ_da = np.load('./Data/dJ_da_expec'+parameters+'.npy')
J_delta_a = delta_a * dJ_da
J_a_cfd = np.load('./Data/J_a_cfd'+parameters+'.npy')


"""Plot CFD vs delta U * dH/dU"""
# plt.plot(times, H_U_cfd, color='red', label='$\\langle\\hat{H}(U+\\frac{\\Delta U}{2})\\rangle  - \\langle\\hat{H}(U-\\frac{\\Delta U}{2})\\rangle$')
# plt.plot(times, H_delta_U, color='blue', linestyle='dashed', label='$\\Delta U \\cdot \\langle \\frac{\\partial \\hat{H}}{\\partial U}\\rangle$')
# plt.ylabel("Energy Expectation")
# plt.title("Energy Expectations at $U = t_0$")
# outfile = "./Plots/H_Taylor_U"+parameters

"""Plot CFD vs delta a  * dH/da"""
# plt.plot(times, H_a_cfd, color='red', label='$\\langle\\hat{H}(a+\\frac{\\Delta a}{2})\\rangle  - \\langle\\hat{H}(a-\\frac{\\Delta a}{2})\\rangle$')
# plt.plot(times, H_delta_a, color='blue', linestyle='dashed', label='$\\Delta a \\cdot \\langle \\frac{\\partial \\hat{H}}{\\partial a}\\rangle$')
# plt.ylabel("Energy Expectation")
# plt.title("Energy Expectations at $U = t_0$")
# outfile = "./Plots/H_Taylor_a"+parameters

"""Plot CFD vs delta U * dJ/dU"""
# plt.plot(times, J_U_cfd, color='red', label='$\\langle\\hat{J}(U+\\frac{\\Delta U}{2})\\rangle  - \\langle\\hat{J}(U-\\frac{\\Delta U}{2})\\rangle$')
# plt.plot(times, J_delta_U, color='blue', linestyle='dashed', label='$\\Delta U \\cdot \\langle \\frac{\\partial \\hat{J}}{\\partial U}\\rangle$')
# plt.ylabel("Current Expectation")
# plt.title("Current Expectations at $U = t_0$")
# outfile = "./Plots/J_Taylor_U"+parameters

"""Plot CFD vs delta a * dJ/da"""
# plt.plot(times, J_a_cfd, color='red', label='$\\langle\\hat{J}(a+\\frac{\\Delta a}{2})\\rangle  - \\langle\\hat{J}(a-\\frac{\\Delta a}{2})\\rangle$')
# plt.plot(times, J_delta_a, color='blue', linestyle='dashed', label='$\\Delta a \\cdot \\langle \\frac{\\partial \\hat{J}}{\\partial a}\\rangle$')
# plt.ylabel("Current Expectation")
# plt.title("Current Expectations at $U = t_0$")
# outfile = "./Plots/J_Taylor_a"+parameters



# plt.legend(loc='upper left')
# plt.xlabel("Time")
# plt.show()
# plt.savefig(outfile+'.pdf')
