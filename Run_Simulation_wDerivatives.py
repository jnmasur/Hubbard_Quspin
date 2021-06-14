##########################################################
# Very basic Hubbard model simulation. Should be used as #
# a base to build on with e.g. interfaces, S.O. coupling #
# tracking and low-rank approximations. Should be fairly #
# self-explanatory, and is based on the Quspin package.  #
# See:  http://weinbe58.github.io/QuSpin/index.html      #
##########################################################
from __future__ import print_function, division
import os
import sys

"""Open MP and MKL should speed up the time required to run these simulations!"""
# threads = sys.argv[1]
threads = 6
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
# line 4 and line 5 below are for development purposes and can be remove
from quspin.operators import hamiltonian, exp_op, quantum_operator  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
import quspin.tools.evolution as evolution
import numpy as np  # general math functions
from scipy.sparse.linalg import eigsh
from time import time  # tool for calculating computation time
import matplotlib.pyplot as plt  # plotting library
import expectations as expec
sys.path.append('../')
from tools import parameter_instantiate as hhg  # Used for scaling units.
import psutil
from evolve import big_evolve, get_phi, evolve_cfd, evolve_psi
# from pyscf import fci
# note cpu_count for logical=False returns the wrong number for multi-socket CPUs.
print("logical cores available {}".format(psutil.cpu_count(logical=True)))
t_init = time()
np.__config__.show()


# def get_ground_state(lat, hop_left, hop_right, onsite):
#     h1 = -lat.t * (hop_left + hop_right)
#     h2 = lat.U * onsite
#     cisolver = fci.direct_spin1.FCI()
#     e, fcivec = cisolver.kernel(h1,h2,lat.nx,(lat.nup,lat.ndown))
#     return e, fcivec.reshape(-1)


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

"""instantiate parameters with proper unit scaling"""
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a, pbc=pbc)

# +- dU/2
lat_plus_dU = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U+(delta_U/2), t=t0, F0=F0, a=a, pbc=pbc)
lat_minus_dU = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U-(delta_U/2), t=t0, F0=F0, a=a, pbc=pbc)

# +- da/2
lat_plus_da = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a+(delta_a/2), pbc=pbc)
lat_minus_da = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a-(delta_a/2), pbc=pbc)

# +da for ffd
lat_plus_U = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U+delta_U, t=t0, F0=F0, a=a, pbc=pbc)

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
stop = cycles / lat.freq
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

"""set up parameters for saving expectations later"""
parameters = '-{}sites-{}t0-{}U-{}delta_U-{}a-{}delta_a-{}field-{}amplitude-{}cycles-{}steps-{}pbc'.format(L,t0,U,delta_U,a,delta_a,field,F0, cycles, n_steps,pbc)

# these change after being scaled so we recalculate
delta_U = lat_plus_dU.U - lat_minus_dU.U
delta_a = lat_plus_da.a - lat_minus_da.a

"""create basis"""
basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))

if lat.U == 0:
    int_list = [[0.0, i, i] for i in range(L)]
else:
    int_list = [[1.0, i, i] for i in range(L)]
static_Hamiltonian_list = [
    ["n|n", int_list] # onsite interaction
]
# n_j,up n_j,down
onsite = hamiltonian(static_Hamiltonian_list, [], basis=basis)

no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
hop = [[1.0, i, i+1] for i in range(L-1)]
if lat.pbc:
    hop.append([1.0, L-1, 0])

# c^dag_j,sigma c_j+1,sigma
hop_left = hamiltonian([["+-|", hop],["|+-", hop]], [], basis=basis, **no_checks)
# c^dag_j+1,sigma c_j,sigma
hop_right = hop_left.getH()


H = -lat.t * (hop_left + hop_right) + lat.U * onsite
H_plus_dU = -lat_plus_dU.t * (hop_left + hop_right) + lat_plus_dU.U * onsite
H_minus_dU = -lat_minus_dU.t * (hop_left + hop_right) + lat_minus_dU.U * onsite

H_plus_U = -lat_plus_U.t * (hop_left + hop_right) + lat_plus_U.U * onsite

"""build ground state"""
print("calculating ground state")
E, psi_0 = H.eigsh(k=1, which='SA')
psi_0 = np.squeeze(psi_0)
psi_0 = psi_0 / np.linalg.norm(psi_0)
print("ground state calculated, energy is {:.2f}".format(E[0]))

"""build ground state for derivatives"""
psi_plus_dU = (H_plus_dU.eigsh(k=1, which='SA'))[1]
psi_minus_dU = (H_minus_dU.eigsh(k=1, which='SA'))[1]
dpsi_dU_0 = (psi_plus_dU-psi_minus_dU)/delta_U
dpsi_dU_0 = np.squeeze(dpsi_dU_0)
dpsi_dU_0 = dpsi_dU_0 / np.linalg.norm(dpsi_dU_0)

dpsi_da_0 = np.zeros((psi_0.shape)[0])

"""Build ground state through imaginary time evolution"""
# E, psi_0 = get_ground_state(lat, hop_left, hop_right, onsite)
# psi_0 = psi_0/np.linalg.norm(psi_0)
# print("ground state calculated, energy is {:.2f}".format(E[0]))
#
# """Ground State for derivatives"""
# psi_plus_dU = (get_ground_state(lat_plus_dU, hop_left, hop_right, onsite))[1]
# psi_minus_dU = (get_ground_state(lat_minus_dU, hop_left, hop_right, onsite))[1]
# dpsi_dU_0 = psi_plus_dU - psi_minus_dU
# dpsi_dU_0 = dpsi_dU_0 / np.linalg.norm(dpsi_dU_0)

# psi_0 = np.load('./Data/psi_0.npy')
# psi_0 = psi_0 / np.linalg.norm(psi_0)
# psi_plus_dU = np.load('./Data/psi_plus_dU.npy')
# psi_minus_dU = np.load('./Data/psi_minus_dU.npy')
# dpsi_dU_0 = np.load('./Data/dpsi_dU_0.npy')
# dpsi_dU_0 = dpsi_dU_0 / np.linalg.norm(dpsi_dU_0)

print('evolving system')
ti = time()
"""evolving system, using our own solver for derivatives"""
# object containing all 3 wavefunctions
big_psi = np.concatenate([psi_0,dpsi_dU_0,dpsi_da_0])

psis_t = evolution.evolve(big_psi, 0.0, times, big_evolve, f_params=(onsite, hop_left, hop_right, lat, cycles))
psis_t = np.squeeze(psis_t)
print("Evolution done! This one took {:.2f} seconds".format(time() - ti))

psi_t = psis_t[:int(len(psis_t)/3)]
dpsi_dU_t = psis_t[int(len(psis_t)/3):int(2*len(psis_t)/3)]
dpsi_da_t = psis_t[int(2*len(psis_t)/3):]

ti = time()
expectations = {}
expectations['H'] = expec.H_expec(psi_t, times, onsite, hop_left, hop_right, lat, cycles)
expectations['J'] = expec.J_expec(psi_t, times, hop_left, hop_right, lat, cycles)
expectations['dH/dU'] = expec.dH_dU_expec(psi_t, dpsi_dU_t, times, onsite, hop_left, hop_right, lat, cycles)
expectations['dH/da'] = expec.dH_da_expec(psi_t, dpsi_da_t, times, onsite, hop_left, hop_right, lat, cycles)
expectations['dJ/dU'] = expec.dJ_dU_expec(psi_t, dpsi_dU_t, times, hop_left, hop_right, lat, cycles)
expectations['dJ/da'] = expec.dJ_da_expec(psi_t, dpsi_da_t, times, hop_left, hop_right, lat, cycles)

print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))

# np.save('./Data/H_expec'+parameters+'.npy', expectations['H'])
# np.save('./Data/J_expec'+parameters+'.npy', expectations['J'])
# np.save('./Data/dH_dU_expec'+parameters+'.npy', expectations['dH/dU'])
# np.save('./Data/dH_da_expec'+parameters+'.npy', expectations['dH/da'])
# np.save('./Data/dJ_dU_expec'+parameters+'.npy', expectations['dJ/dU'])
# np.save('./Data/dJ_da_expec'+parameters+'.npy', expectations['dJ/da'])

"""Evolve cfd for U"""
# psi_plus_dU = psi_plus_dU / np.linalg.norm(psi_plus_dU)
# psi_minus_dU = psi_minus_dU / np.linalg.norm(psi_minus_dU)
# psi_cfd = np.concatenate([psi_plus_dU, psi_minus_dU])
#
# psi_cfd_t = evolution.evolve(psi_cfd, 0.0, times, evolve_cfd, f_params=(onsite, hop_left, hop_right, lat_plus_dU, lat_minus_dU, cycles))
# psi_cfd_t = np.squeeze(psi_cfd_t)
#
# psi_plus_t = psi_cfd_t[:int(len(psi_cfd_t)/2)]
# psi_minus_t = psi_cfd_t[int(len(psi_cfd_t)/2):]
#
# H_U_cfd = expec.H_expec(psi_plus_t, times, onsite, hop_left, hop_right, lat_plus_dU, cycles) - \
#     expec.H_expec(psi_minus_t, times, onsite, hop_left, hop_right, lat_minus_dU, cycles)
#
# J_U_cfd = expec.J_expec(psi_plus_t, times, hop_left, hop_right, lat_plus_dU, cycles) - \
#     expec.J_expec(psi_minus_t, times, hop_left, hop_right, lat_minus_dU, cycles)
#
# np.save('./Data/H_U_cfd'+parameters+'.npy', H_U_cfd)
# np.save('./Data/J_U_cfd'+parameters+'.npy', J_U_cfd)

"""Evolve ffd for U"""
# z, psi_plus_U = H_plus_U.eigsh(k=1, which='SA')
# psi_t = evolution.evolve(psi_plus_U, 0.0, times, evolve_psi, f_params=(onsite, hop_left, hop_right, lat_plus_U, cycles))
# psi_t = np.squeeze(psi_t)
#
# H_plus_U_expec = expec.H_expec(psi_t, times, onsite, hop_left, hop_right, lat_plus_U, cycles)
# np.save('./Data/H_plus_U_expec'+parameters+'.npy', H_plus_U_expec)


"""Evolve cfd for a"""
# ground states for psi+-a = psi
# psi_cfd = np.concatenate([psi_0, psi_0])
#
# psi_cfd_t = evolution.evolve(psi_cfd, 0.0, times, evolve_cfd, f_params=(onsite, hop_left, hop_right, lat_plus_da, lat_minus_da, cycles))
# psi_cfd_t = np.squeeze(psi_cfd_t)
#
# psi_plus_t = psi_cfd_t[:int(len(psi_cfd_t)/2)]
# psi_minus_t = psi_cfd_t[int(len(psi_cfd_t)/2):]
#
# H_a_cfd = expec.H_expec(psi_plus_t, times, onsite, hop_left, hop_right, lat_plus_da, cycles) - \
#     expec.H_expec(psi_minus_t, times, onsite, hop_left, hop_right, lat_minus_da, cycles)
# J_a_cfd = expec.J_expec(psi_plus_t, times, hop_left, hop_right, lat_plus_da, cycles) - \
#     expec.J_expec(psi_minus_t, times, hop_left, hop_right, lat_minus_da, cycles)
#
# np.save('./Data/H_a_cfd'+parameters+'.npy', H_a_cfd)
# np.save('./Data/J_a_cfd'+parameters+'.npy', J_a_cfd)

print('All finished. Total time was {:.2f} seconds using {:d} threads'.format((time() - t_init), threads))

"""This just compares to old data and checks that we're getting the same expectations"""
# old_H_expec = np.load('./Data/H_expec-6sites-3up-3down-0.52t0-0.52U-10cycles-2000steps-Truepbc.npy')
# old_J_expec = np.load('./Data/J_expec-6sites-3up-3down-0.52t0-0.52U-10cycles-2000steps-Truepbc.npy')
#
# H_expec = expectations['H']
# J_expec = expectations['J']
#
# counter = 0
# for i in range(2000):
#     if abs(H_expec[i] - old_H_expec[i]) < abs(old_H_expec[i]) * .01:
#         counter += 1
# print("Percent Energy Expectations accurate:", counter/20)
# counter = 0
# for i in range(2000):
#     if abs(J_expec[i] - old_J_expec[i]) < abs(old_J_expec[i]) * .01:
#         counter += 1
# print("Percent current expectations accurate:", counter/20)
