import numpy as np
from minimization_methods import Parameters, current_expectation_power_spectrum, objective_w_spectrum
from matplotlib import pyplot as plt
from dual_annealing import dual_annealing
get_spectrum = current_expectation_power_spectrum

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

u_over_t_vals = 10 * np.random.random(10)
a_vals = (9 * np.random.random(10)) + 1

pairs = zip(u_over_t_vals, a_vals)

"""
This stuff below has helped me determine that we can discretize
the grid by rounding values to 1e-4 precision. Also, a cost less
than about .25 using this objective function generally indicates
a good agreement between spectra.
"""


# def avg_cost(xs, target_spectrum):
#     """
#     Calculates average cost over each (U/t0, a) pair in xs
#     compared to target_spectrum.
#     """
#     tot_cost = 0
#     for x in xs:
#         x = (x[0] * t0, x[1])
#         spectrum = get_spectrum(x, params)
#         tot_cost += objective_w_spectrum((spectrum, target_spectrum))
#     return tot_cost / len(xs)
#
#
# def compare_spectra(xs, target_spectrum):
#     for x in xs:
#         print("For", x)
#         x = (x[0] * t0, x[1])
#         spectrum = get_spectrum(x, params)
#         cost = objective_w_spectrum((spectrum, target_spectrum))
#         print(cost)
#         plt.semilogy(spectrum)
#         plt.semilogy(target_spectrum, linestyle="dashed")
#         plt.show()
#
#
# for u_over_t, a in pairs:
#     print("Getting target spectrum for ({}, {})".format(u_over_t, a))
#     target = get_spectrum((u_over_t * params.t, a), params)
#     eps = 1e-4
#     comparisons = [(u_over_t + eps, a), (u_over_t - eps, a), (u_over_t + eps, a + eps), (u_over_t - eps, a + eps),
#                    (u_over_t, a + eps), (u_over_t, a - eps), (u_over_t - eps, a + eps), (u_over_t - eps, a - eps)]
#     compare_spectra(comparisons, target)
#     print("Calculating average cost")
#     cost = avg_cost(comparisons, target)
#     print("For epsilon = {}, average cost = {}".format(eps, cost))
#     while cost > 1e-12:
#         eps /= 10
#         comparisons = [(u_over_t + eps, a), (u_over_t - eps, a), (u_over_t + eps, a + eps), (u_over_t - eps, a + eps),
#                        (u_over_t, a + eps), (u_over_t, a - eps), (u_over_t - eps, a + eps), (u_over_t - eps, a - eps)]
#         cost = avg_cost(comparisons, target)
#         print("For epsilon = {}, average cost = {}".format(eps, cost))
#     print("Final epsilon:", eps)
#     print()


def objective(x, p, target_spectrum):
    """
    Objective function designed for dual annealing. Rounds input values
    the fourth decimal place.
    target_spectrum will be used.
    :param x: the candidate (U/t0, a)
    :param p: an instance of the Parameters class
    :param target_spectrum: the spectrum of the target
    """
    x = np.around((x[0] * p.t, x[1]), decimals=4)
    spectrum = get_spectrum(x, p)
    return objective_w_spectrum((spectrum, target_spectrum))


def callback(x, f, context):
    if context == 0:
        ret = "Minimum detected in the annealing process"
    elif context == 1:
        ret = "Detection occurred in the local search process"
    else:
        ret = "Detection done in the dual annealing process"
    print(ret + " at x = {} with fval = {}".format(x, f))


targ_spect = get_spectrum((1 * params. t, 4), params)
bounds = ((0, 10), (1, 10))
res = dual_annealing(objective, bounds, args=(params, targ_spect), maxiter=1000, initial_temp=5230,
                     restart_temp_ratio=2e-5, visit=2.62, accept=-5.0, callback=callback)
