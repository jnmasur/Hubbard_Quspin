from minimization_methods import *
from tools import parameter_instantiate as hhg
import numpy as np
from multiprocessing import Pool
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# radius of range for randomized parameters
rU = 2 * params.t
ra = 2

# bounds for variables
U_upper = 10 * params.t
U_lower = 0
a_upper = 10
a_lower = 0
bounds = ((U_lower, U_upper), (a_lower, a_upper))

threads_per_x0 = 25
num_threads = threads_per_x0 * 1

if __name__ == '__main__':

    target_Us = []
    target_as = []
    runtimes = []
    final_costs = []
    final_Us = []
    final_as = []

    for target_U in np.linspace(1 * t0, 5 * t0, 5):
        for target_a in np.linspace(1, 5, 5):

            ti = time()

            target_Us.append(target_U)
            target_as.append(target_a)

            target_x = np.array([target_U, target_a])
            print()
            print("Target x:", target_x)

            # get J_target
            J_target = current_expectation(target_x, params)

            # array of initial points we will work from
            x0s = np.array([np.array([(U_upper - U_lower) / 4, (a_upper - a_lower) / 4]),
                            np.array([(U_upper - U_lower) / 2, (a_upper - a_lower) / 2]),
                            np.array([3 * (U_upper - U_lower) / 4, 3 * (a_upper - a_lower) / 4])])
            fvals = np.array([objective(x0, J_target, params) for x0 in x0s])

            best_fvals = fvals
            best_xs = x0s

            # gets best fval and x0
            best_ind = np.argmin(fvals)
            best_fval = fvals[best_ind]
            best_x = x0s[best_ind]

            niter = 0
            while best_fval > 1e-9 and niter < 15:
                niter += 1

                # we can use a list to store parameters b/c the parameters do not take up a lot of space
                params_list = []
                for x0 in best_xs:
                    # this line newly added so it will always start at our best xs, leaving the remaining threads
                    # to work with randomized parameters
                    params_list.append((objective, x0, J_target, params, bounds))
                    for seed in get_seeds(threads_per_x0 - 1):
                        params_list.append((objective, randomize_parameters(x0, rU, ra, bounds, seed), J_target, params,
                                            bounds))

                with Pool(num_threads) as pool:

                    n = len(params_list)
                    counter = 0
                    for res in pool.map(minimize_wrapper, params_list):

                        if counter < n // 3:
                            if res.fun < best_fvals[0]:
                                best_fvals[0] = res.fun
                                best_xs[0] = res.x

                        elif counter < 2 * (n // 3):
                            if res.fun < best_fvals[1]:
                                best_fvals[1] = res.fun
                                best_xs[1] = res.x

                        else:
                            if res.fun < best_fvals[2]:
                                best_fvals[2] = res.fun
                                best_xs[2] = res.x

                        counter += 1

                    pool.close()

                best_ind = np.argmin(best_fvals)
                best_fval = best_fvals[best_ind]
                best_x = best_xs[best_ind]

                print("At iteration:", niter)
                print("Best fval: {}, best_x: {}".format(best_fval, best_x))

            tot_time = time() - ti

            runtimes.append(tot_time)
            final_costs.append(best_fval)
            final_Us.append(best_x[0])
            final_as.append(best_x[1])

            print("Final Parameters:", best_x)
            print("Cost function", best_fval)
            print("Total time for pseudo-basinhopping:", tot_time)

    np.save('./GlobalMinimizeData/target_Us-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra), target_Us)
    np.save('./GlobalMinimizeData/target_as-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra), target_as)
    np.save('./GlobalMinimizeData/runtimes-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra), runtimes)
    np.save('./GlobalMinimizeData/final_costs-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra), final_costs)
    np.save('./GlobalMinimizeData/final_Us-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra), final_Us)
    np.save('./GlobalMinimizeData/final_as-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra), final_as)

# target_Us = np.load('./GlobalMinimizeData/target_Us-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra))
# target_Us = np.reshape(target_Us, (5, 5))
# target_as = np.load('./GlobalMinimizeData/target_as-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra))
# target_as = np.reshape(target_as, (5, 5))
# runtimes = np.load('./GlobalMinimizeData/runtimes-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra))
# runtimes = np.reshape(runtimes, (5, 5))
# final_costs = np.load('./GlobalMinimizeData/final_costs-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra))
# final_costs = np.reshape(final_costs, (5, 5))
# final_Us = np.load('./GlobalMinimizeData/final_Us-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra))
# final_Us = np.reshape(final_Us, (5, 5))
# final_as = np.load('./GlobalMinimizeData/final_as-{}threads-{}rU-{}ra.npy'.format(num_threads, rU, ra))
# final_as = np.reshape(final_as, (5, 5))
# U_percent_error = 100 * abs(final_Us - target_Us) / target_Us
# a_percent_error = 100 * abs(final_as - target_as) / target_as
#
# print(U_percent_error)
# print(a_percent_error)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_wireframe(target_Us / t0, target_as, final_costs)
# plt.xlabel("Target U/t0")
# plt.ylabel("Target a")
# plt.title("Cost")
# plt.show()
