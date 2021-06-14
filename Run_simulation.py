##########################################################
# Very basic Hubbard model simulation. Should be used as #
# a base to build on with e.g. interfaces,
# n/index.html      #
##########################################################

from quspin.operators import hamiltonian  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
import quspin.tools.evolution as evolution
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
import expectations as expec
from tools import parameter_instantiate as hhg  # Used for scaling units.
from evolve import evolve_psi


t_init = time()

"""Hubbard model Parameters"""
L = 12  # system size
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

"""instantiate parameters with proper unit scaling"""
lat = hhg(field=field, nup=N_up, ndown=N_down, nx=L, ny=0, U=U, t=t0, F0=F0, a=a, pbc=pbc)

"""System Evolution Time"""
cycles = 10  # time in cycles of field frequency
n_steps = 2000
start = 0
stop = cycles / lat.freq
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

"""set up parameters for saving expectations later"""
parameters = f'-{L}sites-{t0}t0-{U}U-{a}a-{field}field-{F0}amplitude-{cycles}cycles-{n_steps}steps-{pbc}pbc'

"""create basis"""
basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down), sblock=1, kblock=1)

"""Create static part of hamiltonian - the interaction b/w electrons"""
int_list = [[1.0, i, i] for i in range(L)]
static_Hamiltonian_list = [
    ["n|n", int_list]  # onsite interaction
]
# n_j,up n_j,down
onsite = hamiltonian(static_Hamiltonian_list, [], basis=basis)

"""Create dynamic part of hamiltonian - composed of a left and a right hopping parts"""
hop = [[1.0, i, i+1] for i in range(L-1)]
if lat.pbc:
    hop.append([1.0, L-1, 0])
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
# c^dag_j,sigma c_j+1,sigma
hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=basis, **no_checks)
# c^dag_j+1,sigma c_j,sigma
hop_right = hop_left.getH()

"""Create complete Hamiltonian"""
H = -lat.t * (hop_left + hop_right) + lat.U * onsite

"""get ground state as the eigenstate corresponding to the lowest eigenergy"""
print("calculating ground state")
E, psi_0 = H.eigsh(k=1, which='SA')
psi_0 = np.squeeze(psi_0)
psi_0 = psi_0 / np.linalg.norm(psi_0)
print("ground state calculated, energy is {:.2f}".format(E[0]))

print('evolving system')
ti = time()
"""evolving system, using our own solver for derivatives"""
psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=(onsite, hop_left, hop_right, lat, cycles))
psi_t = np.squeeze(psi_t)
print("Evolution done! This one took {:.2f} seconds".format(time() - ti))

"""Calculate Expectation Values"""
ti = time()
expectations = {'H': expec.H_expec(psi_t, times, onsite, hop_left, hop_right, lat, cycles),
                'J': expec.J_expec(psi_t, times, hop_left, hop_right, lat, cycles)}

print("Expectations calculated! This took {:.2f} seconds".format(time() - ti))

np.save('./Data/H_expec'+parameters+'.npy', expectations['H'])
np.save('./Data/J_expec'+parameters+'.npy', expectations['J'])

print('All finished. Total time was {:.2f} seconds'.format((time() - t_init)))
