from __future__ import print_function, division
import os
import sys

"""Open MP and MKL should speed up the time required to run these simulations!"""
# threads = sys.argv[1]
threads = 6
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
from scipy.optimize import minimize
from old_minimization_methods import objective, J_gradient, Parameters


sys.path.append('../')
import psutil
# note cpu_count for logical=False returns the wrong number for multi-socket CPUs.
print("logical cores available {}".format(psutil.cpu_count(logical=True)))
t_init = time()
np.__config__.show()

"""Hubbard model Parameters"""
L = 6 # system size
N_up = L // 2 + L % 2  # number of fermions with spin up
N_down = L // 2  # number of fermions with spin down
N = N_up + N_down  # number of particles
t0 = 0.52  # hopping strength
pbc = True

"""Laser pulse parameters"""
field = 32.9  # field angular frequency THz
F0 = 10  # Field amplitude MV/cm

"""delta U/a for finding derivatives"""
delta_U = .001
delta_a = .001

"""Initial guesses for optimization parameters"""
U = 1.3 * t0
a = 4.5

# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, delta_U, delta_a, pbc)
params.set_basis()

"""Initial parameters for optimization"""
x0 = np.array([U, a])

"""The target current"""
J_target = np.load('./Data/J_expec-6sites-0.52t0-0.52U-0.1delta_U-4a-0.001delta_a-32.9field-10amplitude-10cycles-2000steps-Truepbc.npy')

ti = time()
"""The actual minimization call"""
# z = minimize(objective, x0, args=(J_target, params), jac=J_gradient, bounds=((0,10),(0,10)), options={'disp': True})
z = minimize(objective, x0, args=(J_target, params), bounds=((0,10),(0,10)), options={'disp': True})

print(z.x)

objective(z.x,J_target,params,graph=True)

print("Total time for optimization:", time()-ti)
