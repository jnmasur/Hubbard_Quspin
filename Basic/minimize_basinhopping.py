from scipy.optimize import basinhopping
from old_minimization_methods import Parameters, objective, end_minimization
import numpy as np
from time import time

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
U = 1.05 * t0
a = 4.05

# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, delta_U, delta_a, pbc)
params.set_basis()

parameters = '-{}sites-{}t0-{}field-{}amplitude-{}cycles-{}steps-{}pbc'.format(params.nx,params.t, params.field,params.F0,10,2000,params.pbc)

# initial guess point
x0 = np.array([U,a])

# target current
J_target = np.load('./Data/J_expec-6sites-0.52t0-0.52U-0.1delta_U-4a-0.001delta_a-32.9field-10amplitude-10cycles-2000steps-Truepbc.npy')

initial_U = []
initial_a = []
U_percent_error = []
a_percent_error = []
runtime = []
for i in range(20):
    U = (i+1) * .1 * t0
    a = (i+1) * .4

    initial_U.append(U)
    initial_a.append(a)

    x0 = np.array([U,a])

    ti = time()
    z = basinhopping(objective, x0, niter=35,
                     minimizer_kwargs={'args': (J_target, params), 'bounds': ((0, None), (0, None)),
                                       'method': "L-BFGS-B", 'options': {'ftol': 10 ** -8, 'maxiter': 50}}, disp=True,
                     callback=end_minimization)
    tot_time = time() - ti
    print("Total time for Basin-Hopping", tot_time)
    runtime.append(tot_time)

    print(z.x)
    U_err = 100*(z.x[0] - .52)/.52
    a_err = 100*(z.x[1] - 4)/4

    U_percent_error.append(U_err)
    a_percent_error.append(a_err)

np.save('./BasinHoppingData/initial_U'+parameters+'.npy', initial_U)
np.save('./BasinHoppingData/initial_a'+parameters+'.npy', initial_a)
np.save('./BasinHoppingData/runtimes'+parameters+'.npy', runtime)
np.save('./BasinHoppingData/U_percent_error'+parameters+'.npy', U_percent_error)
np.save('./BasinHoppingData/a_percent_error'+parameters+'.npy', a_percent_error)

# print(z.x)
#
# objective(z.x,J_target,params,graph=True,x0=x0)
