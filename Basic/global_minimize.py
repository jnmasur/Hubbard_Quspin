from minimization_methods import objective, Parameters, randomize_parameters, get_seeds, minimize_wrapper
from tools import parameter_instantiate as hhg
import numpy as np
from multiprocessing import Pool
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
initial_U = 2 * t0
initial_a = 3

# initial guess point
x0 = np.array([initial_U, initial_a])

# target parameters
target_U = 1 * t0
target_a = 4

lat = hhg(field, N_up, N_down, L, 0, target_U, t0, F0=F0, a=target_a, pbc=pbc)
cycles = 10
n_steps = 2000
start = 0
stop = cycles / lat.freq
target_delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)[1]
# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, target_delta, pbc)
params.set_basis()

parameters = f'-{params.nx}sites-{params.t}t0-{target_U}U-{target_a}a-{params.field}field-' \
             f'{params.F0}amplitude-{10}cycles-{2000}steps-{params.pbc}pbc'

# target current
J_target = np.load('./Data/J_expec' + parameters + '.npy')

# radius of range for randomized parameters
rU = .4 * params.t
ra = .4

# bounds for variables
U_upper = 10 * params.t
U_lower = 0
a_upper = 10
a_lower = 0
bounds = ((U_lower, U_upper), (a_lower, a_upper))

if __name__ == '__main__':

    # get the first value of the function
    fval = objective(x0, J_target, params)
    best_fval = fval
    best_x = x0

    niter = 0
    while best_fval > 1e-6 and niter < 20:
        niter += 1

        # create an iterator of parameters to be passed into minimize with randomized x0
        iter_params = ((objective, randomize_parameters(best_x, rU, ra, bounds, seed), J_target, params, bounds)
                       for seed in get_seeds(100))

        with Pool() as pool:

            # minimize with different initial parameters then compare to find the best x0
            for res in pool.imap_unordered(minimize_wrapper, iter_params):
                if res.fun < best_fval:
                    best_fval = res.fun
                    best_x = res.x

pass
# initial_Us = []
# initial_as = []
# runtimes = []
# fvals = []
# final_Us = []
# final_as = []
#
# for initial_U in np.linspace(target_U/3, 5*target_U/3, 5):
#     for initial_a in np.linspace(target_a/3, 5*target_a/3, 5):
#         x0 = np.array([initial_U, initial_a])
#
#         initial_Us.append(initial_U)
#         initial_as.append(initial_a)
#
#         ti = time()
#         res = global_minimize(x0, objective, J_target, params)
#         tot_time = time() - ti
#
#         runtimes.append(tot_time)
#         fvals.append(res.fun)
#         final_Us.append(res.x[0])
#         final_as.append(res.x[1])
#
# np.save('./GlobalMinimizeData/initial_Us'+parameters+'.npy', initial_Us)
# np.save('./GlobalMinimizeData/initial_as'+parameters+'.npy', initial_as)
# np.save('./GlobalMinimizeData/runtimes'+parameters+'.npy', runtimes)
# np.save('./GlobalMinimizeData/cost'+parameters+'.npy', fvals)
# np.save('./GlobalMinimizeData/final_Us'+parameters+'.npy', final_Us)
# np.save('./GlobalMinimizeData/final_as'+parameters+'.npy', final_as)
