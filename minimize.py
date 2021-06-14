from minimization_methods import objective, Parameters, minimize_wrapper
from tools import parameter_instantiate as hhg
import numpy as np
from time import time

"""Hubbard model Parameters"""
L = 6  # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
pbc = True

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm

"""Initial guesses for optimization parameters"""
initial_U = 1.05 * t0
initial_a = 4.05

# initial guess point
x0 = np.array([initial_U, initial_a])

# target parameters
target_U = 1 * t0
target_a = 4

# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, pbc)
params.set_basis()

parameters = f'-{params.nx}sites-{params.t}t0-{target_U}U-{target_a}a-{params.field}field-' \
             f'{params.F0}amplitude-{10}cycles-{2000}steps-{params.pbc}pbc'

# target current
J_target = np.load('./Data/J_expec' + parameters + '.npy')

ti = time()
"""The actual minimization call"""
z = minimize_wrapper((objective, x0, J_target, params, ((0,10*params.t),(0,10))))

print(z.x)
print(z.fun)
print(z.message)

objective(z.x,J_target,params,graph=True)

print("Total time for optimization:", time()-ti)
