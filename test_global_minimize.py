from minimization_methods import *
from tools import parameter_instantiate as hhg
import numpy as np
from multiprocessing import cpu_count, Pool
from time import time
from warnings import warn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

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

# add all parameters to the class and create the basis
params = Parameters(L, N_up, N_down, t0, field, F0, pbc)
params.set_basis()

"""Parameters for basinhopping_runner"""
# radius of range for randomized parameters
rU = 2 * params.t
ra = 2

# bounds for variables
U_upper = 10 * params.t
U_lower = 0
a_upper = 10
a_lower = 0
bounds = ((U_lower, U_upper), (a_lower, a_upper))

threads_per_x0 = 7

num_x0s = 15
num_threads = threads_per_x0 * num_x0s

reducs = 2  # number of reductions
r_fctr = None  # how we split the number of initial x0s
r_absolute = 5  # cuts down x0s by a constant
adj_rs = True

maxiter = 30

parameters = "-{}threadsperx0-{}numx0s-{}rU-{}ra-{}maxiter-{}reductions-{}r_factor-{}r_abs-{}adjust_radii".format(
    threads_per_x0, num_x0s, rU, ra, maxiter, reducs, r_fctr, r_absolute, adj_rs)
# parameters = "-{}threadsperx0-{}numx0s-{}rU-{}ra-{}maxiter-{}reductions-{}r_factor-{}r_abs".format(
#     threads_per_x0, num_x0s, rU, ra, maxiter, reducs, r_fctr, r_absolute)
print("Parameters for this run:\n{}".format(parameters))

# if __name__ == '__main__':
#
#     def _get_initial(target_current, prms, n_x0s, thread_count, bnds):
#         """
#         :return: the initial x0s and fvals for basinhopping
#         """
#         # x0s and fvals are initialized as arrays of infinity
#         fvals = np.array([np.inf for _ in range(n_x0s)])
#         x0s = np.array([np.array([np.inf, np.inf]) for _ in range(n_x0s)])
#
#         while max(fvals) == np.inf:
#             # these temporary parameters consist of num_threads initial points that are randomized anywhere in bounds
#             temp_params = [(objective, randomize_parameters(np.array([(bnds[0][1] - bnds[0][0]) / 2,
#                                                                       (bnds[1][1] - bnds[1][0]) / 2]),
#                                                             bnds[0][1], bnds[1][1], bnds, seed), target_current, prms,
#                             bnds) for seed in get_seeds(thread_count)]
#
#             # find the best initial points that are not too close to one another
#             with Pool(thread_count) as pool:
#
#                 for result in pool.imap_unordered(minimize_wrapper, temp_params):
#                     add = True
#                     replace_ind = np.argmax(fvals)
#                     if result.fun < fvals[replace_ind]:
#                         if abs(np.linalg.norm(result.x) - np.linalg.norm(x0s[replace_ind])) < 1:
#                             add = False
#
#                         if add:
#                             x0s[replace_ind] = result.x
#                             fvals[replace_ind] = result.fun
#
#                 pool.close()
#
#         return x0s, fvals
#
#     def _reduce_xs(new_num_x0s, old_x0s, old_fvals):
#         """
#         Reduces the number of initial points to be used
#         :param new_num_x0s: the number of initial points we now want
#         :param old_x0s: the old best_xs
#         :param old_fvals: the old best_fvals
#         :return: the new best_xs and fvals
#         """
#         new_x0s = []
#         new_fvals = []
#         for i in range(new_num_x0s):
#             ind = np.argmin(old_fvals)  # find the smallest function value
#             new_x0s.append(old_x0s[ind])  # add the current best x0 to the new array
#             new_fvals.append(old_fvals[ind])  # add the current best fval to the new array
#             old_fvals[ind] = np.inf  # replace the lowest value with infinity so we dont use it again
#         new_x0s = np.array(new_x0s)
#         new_fvals = np.array(new_fvals)
#         thrds = cpu_count() // new_num_x0s  # calculate new number of threads per x0
#         print("New x values:")
#         print(new_x0s)
#         return new_x0s, new_fvals, thrds
#
#     def _adjust_radii(new_num_x0s, old_num_x0s, old_rU, old_ra):
#         """
#         Adjusts the radii of randomization
#         :param new_num_x0s: the number of x0s that we are reducing to
#         :param old_num_x0s: the previous number of x0s
#         :return: new_rU, new_ra
#         """
#         factor = old_num_x0s / new_num_x0s
#         return old_rU * factor, old_ra * factor
#
#     def basinhopping_runner(target_current, prms, thrds_per_x0, n_x0s, U_rad, a_rad, bnds, ftol=1e-9, max_iters=20,
#                             reductions=0, r_factor=None, r_abs=None, adjust_radii = False):
#         """
#         Runs Basinhopping on multiple threads
#         :param target_current: the target current, currently not the power series
#         :param prms: an instance of Parameters
#         :param thrds_per_x0: number of threads we will use per x0
#         :param n_x0s: number of initial points to work from
#         :param U_rad: radius for U randomization
#         :param a_rad: radius for a randomization
#         :param bnds: the bounds for the variables
#         :param ftol: tolerance for termination of algorithm
#         :param max_iters: the maximum number of iterations allowed
#         :param reductions: number of reductions in x0s (max_iters % (reductions+1) = 0)
#         :param r_factor: if not None, reduce by this factor every reduction (num_x0s -> num_x0s/r_factor)
#         :param r_abs: if not None, reduce num_x0s by this number every reduction (num_x0s -> num_x0s - r_abs)
#         :param adjust_radii: if True, increase radii size with reductions in number of x0s
#         :return: best_x, cost, runtime
#         """
#         ti = time()
#
#         thread_count = n_x0s * thrds_per_x0
#         assert thread_count < cpu_count(), "Too many threads!"
#         assert max_iters % (reductions + 1) == 0, "Maximum iterations do not split evenly into reductions"
#         if r_factor is not None and r_abs is not None:
#             raise Exception("Cannot reduce by factor and absolute number at the same time")
#         if r_factor is not None:
#             assert n_x0s / (r_factor ** reductions) >= 1, "Too many reductions/reduction factor too large"
#         if r_abs is not None:
#             assert n_x0s - (reductions * r_abs) >= 1, "Too many reductions/reduction decrease too large"
#         if adjust_radii and reductions == 0:
#             warn("Radii will not be adjusted no reductions are to occur")
#
#         x0s, fvals = _get_initial(target_current, prms, n_x0s, thread_count, bnds)
#         best_fvals = fvals
#         best_xs = x0s
#         print("Initial x values:")
#         print(x0s)
#
#         # gets best fval and x0
#         best_ind = np.argmin(fvals)
#         best_fval = fvals[best_ind]
#         best_x = x0s[best_ind]
#
#         niter = 0
#         riter = 0  # keeps track of the number of iterations within a reduction loop
#         if reductions > 0:
#             max_riters = max_iters // (reductions + 1)
#         else:
#             max_riters = np.inf
#
#         while best_fval > ftol and niter < max_iters:
#             if riter >= max_riters:
#                 riter = 0
#                 if r_factor is not None:
#                     new_num_x0s = n_x0s // r_factor  # calculates new number of x0s
#                     U_rad, a_rad = _adjust_radii(new_num_x0s, num_x0s, U_rad, a_rad)
#                     best_xs, best_fvals, thrds_per_x0 = _reduce_xs(new_num_x0s, best_xs, best_fvals)
#                     n_x0s = new_num_x0s
#                 elif r_abs is not None:
#                     new_num_x0s = n_x0s - r_abs
#                     U_rad, a_rad = _adjust_radii(new_num_x0s, num_x0s, U_rad, a_rad)
#                     best_xs, best_fvals, thrds_per_x0 = _reduce_xs(new_num_x0s, best_xs, best_fvals)
#                     n_x0s = new_num_x0s
#                 else:
#                     raise Exception("Tried to reduce x0s, but r_factor and r_abs were both None")
#
#             riter += 1
#             niter += 1
#
#             # we can use a list to store parameters b/c the parameters do not take up a lot of space
#             params_list = []
#             for x0 in best_xs:
#                 # this line added so it will always start at our best xs, leaving the remaining threads
#                 # to work with randomized parameters
#                 params_list.append((objective, x0, target_current, prms, bnds))
#                 for seed in get_seeds(thrds_per_x0 - 1):
#                     params_list.append((objective, randomize_parameters(x0, U_rad, a_rad, bnds, seed), target_current,
#                                         prms, bnds))
#
#             with Pool(thread_count) as pool:
#
#                 counter = 0
#                 for result in pool.map(minimize_wrapper, params_list):
#                     index = counter // thrds_per_x0
#
#                     if result.fun < best_fvals[index]:
#                         best_fvals[index] = result.fun
#                         best_xs[index] = result.x
#
#                     counter += 1
#
#                 pool.close()
#
#             best_ind = np.argmin(best_fvals)
#             best_fval = best_fvals[best_ind]
#             best_x = best_xs[best_ind]
#
#             print("At iteration:", niter)
#             print("Best fval: {}, best_x: {}".format(best_fval, best_x))
#
#         tot_time = time() - ti
#
#         return Result(best_x, best_fval, tot_time)
#
#     target_Us = []
#     target_as = []
#     runtimes = []
#     final_costs = []
#     final_Us = []
#     final_as = []
#
#     # for target_U in np.linspace(1 * t0, 5 * t0, 5):
#     #     for target_a in np.linspace(1, 5, 5):
#     for target_U in np.linspace((1/3) * t0, (5/3) * t0, 5):
#         for target_a in np.linspace(1, 5, 5):
#
#             target_Us.append(target_U)
#             target_as.append(target_a)
#
#             target_x = np.array([target_U, target_a])
#             print()
#             print("Target x:", target_x)
#
#             # get J_target
#             J_target = current_expectation_power_spectrum(target_x, params)
#
#             res = basinhopping_runner(J_target, params, threads_per_x0, num_x0s, rU, ra, bounds, max_iters=maxiter,
#                                       reductions=reducs, r_factor=r_fctr, r_abs=r_absolute, adjust_radii=adj_rs)
#
#             runtimes.append(res.time)
#             final_costs.append(res.fun)
#             final_Us.append(res.x[0])
#             final_as.append(res.x[1])
#
#             print("Final Parameters:", res.x)
#             print("Cost function", res.fun)
#             print("Total time for pseudo-basinhopping:", res.time)
#
#     np.save('./GlobalMinimizeData/target_Us'+parameters+'.npy', target_Us)
#     np.save('./GlobalMinimizeData/target_as'+parameters+'.npy', target_as)
#     np.save('./GlobalMinimizeData/runtimes'+parameters+'.npy', runtimes)
#     np.save('./GlobalMinimizeData/final_costs'+parameters+'.npy', final_costs)
#     np.save('./GlobalMinimizeData/final_Us'+parameters+'.npy', final_Us)
#     np.save('./GlobalMinimizeData/final_as'+parameters+'.npy', final_as)

target_Us = np.load('./GlobalMinimizeData/target_Us'+parameters+'.npy')
target_Us = np.reshape(target_Us, (5, 5))
target_as = np.load('./GlobalMinimizeData/target_as'+parameters+'.npy')
target_as = np.reshape(target_as, (5, 5))
runtimes = np.load('./GlobalMinimizeData/runtimes'+parameters+'.npy')
tot_time = sum(runtimes) / 3600
runtimes = np.reshape(runtimes, (5, 5))
final_costs = np.load('./GlobalMinimizeData/final_costs'+parameters+'.npy')
final_costs = np.reshape(final_costs, (5, 5))
final_Us = np.load('./GlobalMinimizeData/final_Us'+parameters+'.npy')
final_Us = np.reshape(final_Us, (5, 5))
final_as = np.load('./GlobalMinimizeData/final_as'+parameters+'.npy')
final_as = np.reshape(final_as, (5, 5))
U_percent_error = 100 * abs(final_Us - target_Us) / target_Us
a_percent_error = 100 * abs(final_as - target_as) / target_as

# avg_U_err = np.mean(U_percent_error)
# avg_a_error = np.mean(a_percent_error)
#
# U_correct = sum([1 for x in U_percent_error.flatten() if x <= 10])
# a_correct = sum([1 for x in a_percent_error.flatten() if x <= 10])
#
# print("U correct:", U_correct)
# print("a correct:", a_correct)
#
# print("Average U error:", avg_U_err)
# print("Average a error:", avg_a_error)
#
# print("Time in hours:", tot_time)
#
# table = {"Target U": target_Us.flatten(), "Target a": target_as.flatten(), "Cost": final_costs.flatten(),
#          "U % error": U_percent_error.flatten(), "a % error": a_percent_error.flatten()}
# print(tabulate(table, headers="keys"))


"""This is an attempt to graph them all on the same figure, it doesnt work well"""
# def plot_spectrum_comparisons(ax, U_ind, a_ind):
#     result_U = final_Us[U_ind, a_ind]
#     result_a = final_as[U_ind, a_ind]
#     result_x = np.array([result_U, result_a])
#     targ_U = target_Us[U_ind, a_ind]
#     targ_a = target_as[U_ind, a_ind]
#     targ_x = np.array([targ_U, targ_a])
#
#     result_J = current_expectation_power_spectrum(result_x, params)
#     targ_J = current_expectation_power_spectrum(targ_x, params)
#
#     ax.set_title("U error = {:.3f}%, a error = {:.3f}%".format(U_percent_error[U_ind, a_ind], a_percent_error[U_ind, a_ind]))
#     # ax.title("Comparing Power Spectrums")
#     ax.semilogy(result_J, label="Result: $U=%.2f\\frac{U}{t0}$, $a=%.2f$" % (result_U / t0, result_a), color='blue')
#     ax.semilogy(targ_J, label="targ: $U=%.2f\\frac{U}{t0}$, $a=%.2f$" % (targ_U / t0, targ_a), color='red',
#             linestyle='dashed')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.legend(loc='upper right')
#
#
# fig, axs = plt.subplots(2, 2)
# fig.figsize = (12, 9)
# plot_spectrum_comparisons(axs[0, 0], 0, 0)
# plot_spectrum_comparisons(axs[0, 1], 4, 0)
# plot_spectrum_comparisons(axs[1, 0], 0, 4)
# plot_spectrum_comparisons(axs[1, 1], 4, 4)

"""Plotting the various spectrums against each other"""
U_ind = 4
a_ind = 0

result_U = final_Us[U_ind, a_ind]
result_a = final_as[U_ind, a_ind]
result_x = np.array([result_U, result_a])
targ_U = target_Us[U_ind, a_ind]
targ_a = target_as[U_ind, a_ind]
targ_x = np.array([targ_U, targ_a])

result_J = current_expectation_power_spectrum(result_x, params)
targ_J = current_expectation_power_spectrum(targ_x, params)

plt.title("U error = {:.3f}%, a error = {:.3f}%".format(U_percent_error[U_ind, a_ind], a_percent_error[U_ind, a_ind]))
plt.semilogy(result_J, label="Result: $U=%.2f\\frac{U}{t0}$, $a=%.2f$" % (result_U / t0, result_a), color='blue')
plt.semilogy(targ_J, label="targ: $U=%.2f\\frac{U}{t0}$, $a=%.2f$" % (targ_U / t0, targ_a), color='red',
        linestyle='dashed')
# ax.set_xticklabels([])
# ax.set_yticklabels([])
plt.legend(loc='upper right')

plt.savefig('./GlobalMinimizeData/spectrum_comparison-{}U_index-{}a_index.png'.format(U_ind, a_ind))
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_wireframe(target_Us / t0, target_as, final_costs)
# plt.xlabel("Target U/t0")
# plt.ylabel("Target a")
# plt.title("Cost")
# # plt.savefig('./GlobalMinimizeData/cost_plot'+parameters+'.png')
# plt.show()
