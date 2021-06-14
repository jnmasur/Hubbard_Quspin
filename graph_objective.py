from minimization_methods import *
from tools import parameter_instantiate as hhg
import numpy as np
from multiprocessing import Pool
from time import time
from warnings import warn
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
# from tabulate import tabulate
from matplotlib.colors import LogNorm


def set_up(u_over_t, a, symmetry, sites):
    """
    Sets up system
    :returns: the target x and parameters
    """

    # Hubbard model Parameters
    L = sites  # system size
    N_up = L // 2 + L % 2  # number of fermions with spin up
    N_down = L // 2  # number of fermions with spin down
    N = N_up + N_down  # number of particles
    t0 = 0.52  # hopping strength
    pbc = True

    # Laser pulse parameters
    field = 32.9  # field angular frequency THz
    F0 = 10  # Field amplitude MV/cm

    # target parameters
    target_U = u_over_t * t0
    target_a = a
    target_x = np.array([target_U, target_a])

    # add all parameters to the class and create the basis
    params = Parameters(L, N_up, N_down, t0, field, F0, pbc)
    params.set_basis(symmetry)

    return target_x, params


def get_data_points(params, U_over_t0_lower=0, U_over_t0_upper=10, a_lower=0, a_upper=10):
    """
    :returns: list of data points that form a grid over (U_lower, U_upper) * (a_lower, a_upper), bounds, U_vals, a_vals
    """

    U_lower = params.t * U_over_t0_lower
    U_upper = params.t * U_over_t0_upper

    bounds = ((U_lower, U_upper), (a_lower, a_upper))

    U_vals = np.linspace(U_lower, U_upper, num=100)
    a_vals = np.linspace(a_lower, a_upper, num=100)

    U_vals, a_vals = np.meshgrid(U_vals, a_vals)
    U_vals2 = list(U_vals.ravel())
    a_vals2 = list(a_vals.ravel())

    x_vals = list(zip(U_vals2, a_vals2))

    return x_vals, bounds, U_vals, a_vals


def get_spectra(params, x_vals, parameters):
    """
    Saves spectra over a grid for processing
    :return: the spectra
    """
    # the lat is for getting the pulse frequency so it does not depend on U
    lat = hhg(params.field, params.nup, params.ndown, params.nx, 0, 0, params.t)
    data_points = ((np.array(point), params) for point in x_vals)

    if __name__ == "__main__":
        # this saves the spectra for processing
        with Pool(100) as pool:
            results = pool.starmap(current_expectation_power_spectrum, data_points)
        results = np.array(results)
        print(results.shape)

        # scale frequencies in terms of omega0
        scaled_results = []
        for pair in results:
            freqs, spect = pair
            freqs = freqs / lat.freq
            scaled_results.append((freqs, spect))
        scaled_results = np.array(scaled_results)
        print(scaled_results.shape)
        np.save("./Spectra/spectra" + parameters + ".npy", scaled_results)

        return scaled_results


def get_current_expecs(params, x_vals, parameters):
    """
    Saves and returns the current expectations over a grid of points
    """
    data_points = ((np.array(point), params) for point in x_vals)

    if __name__ == "__main__":
        with Pool(100) as pool:
            results = pool.starmap(current_expectation, data_points)
        results = np.array(results)

        np.save("./CurrentExpectations/CurrentExpectations" + parameters + ".npy", results)
        return results


def slice_spectra(spectra, parameters, min_cut, max_cut):
    """
    Slices spectra between min_cut and max_cut which are numbers in terms of the driving
    frequency.
    :return: zipped frequencies and spectra and indices for cutting
    """

    new_spectra = []
    cut_indices = []
    for freqs, spect in spectra:
        min_indx = np.searchsorted(freqs, min_cut)
        max_indx = np.searchsorted(freqs, max_cut)
        spect = spect[min_indx:max_indx]
        freqs = freqs[min_indx:max_indx]
        new_spectra.append((freqs, spect))
        cut_indices.append((min_indx, max_indx))

    spectrum_parameters = parameters[parameters.index('-sites'):]
    np.save('./Spectra/cutoffSpectra' + spectrum_parameters + '.npy', new_spectra)
    np.save('./Spectra/CutIndices' + spectrum_parameters + '.npy', cut_indices)

    return np.array(new_spectra), np.array(cut_indices)


def get_cost(params, target_x, spectra, cut_indices):
    target_freqs, target_current = current_expectation_power_spectrum(target_x, params)
    min_indx, max_indx = cut_indices[0]
    target_current = target_current[min_indx:max_indx]

    costs = np.array([objective_w_spectrum((target_current, spec)) for _, spec in spectra])
    return costs


def plot_cost(costs, parameters, params, target_x, U_vals, a_vals, type='2d', add_title=False):
    target_U, target_a = target_x
    if costs is None:
        costs = np.load("./CostData/Costs" + parameters + ".npy")
    costs = np.reshape(costs, (100, 100))

    if type == '2d':
        plt.imshow(
            costs,
            origin='lower',
            extent=[U_vals.min() / params.t, U_vals.max() / params.t, a_vals.min(), a_vals.max()]
        )
        plt.plot([target_U / params.t], [target_a], '+r')
        plt.colorbar(label="Cost")

        plt.xlabel("U/t0")
        plt.ylabel("a")
        if add_title:
            title = "Cost for U = {}t_0 and a = {}".format(target_U / params.t, target_a)
            plt.title(title)
        plt.savefig('./CostData/plot' + parameters + '.png')
        # plt.show()
        plt.close()

    # NOT FULLY IMPLEMENTED DONT USE THIS YET
    elif type == '3d':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(U_vals / params.t, a_vals, costs, cmap=cm.get_cmap('coolwarm'))
        plt.show()


def plot_response(params, xs, domain):
    """
    Plots material response based on their Hubbard parameters.
    :param xs: list (or iterable) of points from which we can calculate the response
    :param domain: either "time" or "frequency", determines whether we plot in the spectra or actual response
    """

    if domain == "time":
        outfile = "./DirectComparisons/TimeDomainComparison"
        for x in xs:
            times, current = current_expectation(x, params, return_time=True)
            plt.plot(times, current, label="$U = {} \cdot t_0, a = {}$".format(x[0] / params.t, x[1]),
                     color=(np.random.random(), np.random.random(), np.random.random()))
            outfile += "-({},{})".format(x[0], x[1])
        plt.xlabel("Time")
        plt.ylabel("Current Density Expectation")
        plt.legend(loc="upper right")
        plt.savefig(outfile + ".png")
        plt.show()
    elif domain == "frequency":
        # the lat is for getting the pulse frequency so it does not depend on U
        lat = hhg(params.field, params.nup, params.ndown, params.nx, 0, 0, params.t)
        fig, ax = plt.subplots()
        outfile = "./DirectComparisons/FrequencyDomainComparison"
        for x in xs:
            freqs, spect = current_expectation_power_spectrum(x, params)
            ax.semilogy(freqs / lat.freq, spect, label="$U = {} \cdot t_0, a = {}$".format(x[0] / params.t, x[1]),
                        color=(np.random.random(), np.random.random(), np.random.random()))
            outfile += "-({},{})".format(x[0], x[1])
        ax.set_xlabel("Harmonic Order")
        ax.set_ylabel("Power")
        # ax.set_xlim((0, 50))
        for i in range(1, 100, 2):
            ax.axvline(i, linestyle="dashed", color="grey")
        ax.legend(loc="upper right")
        plt.savefig(outfile + ".png")
        plt.show()
    else:
        raise Exception("Invalid specifier for domain")


def compare_response(params, x1, x2, domain, normalized=False):
    """
    Compares the response of two materials
    :param domain: either "time" or "frequency", determines whether we plot in the spectra or actual response
    """

    if domain == "time":
        outfile = "./DirectComparisons/TimeDomainComparison-({},{})-({},{})".format(x1[0], x1[1], x2[0], x2[1])
        times, current1 = current_expectation(x1, params, return_time=True)
        current2 = current_expectation(x2, params)
        if normalized:
            current1 /= current1.max()
            current2 /= current2.max()
        plt.plot(times, current1, label="$U = {} \cdot t_0, a = {}$".format(x1[0] / params.t, x1[1]), color="blue")
        plt.plot(times, current2, label="$U = {} \cdot t_0, a = {}$".format(x2[0] / params.t, x2[1]), color="red",
                 linestyle="dashed")
        plt.xlabel("Time")
        plt.ylabel("Current Density Expectation")
        plt.legend(loc="upper right")
        plt.savefig(outfile + ".png")
        plt.show()
    elif domain == "frequency":
        # the lat is for getting the pulse frequency so it does not depend on U
        lat = hhg(params.field, params.nup, params.ndown, params.nx, 0, 0, params.t)
        fig, ax = plt.subplots()
        outfile = "./DirectComparisons/FrequencyDomainComparison-({},{})-({},{})".format(x1[0], x1[1], x2[0], x2[1])
        freqs, spect1 = current_expectation_power_spectrum(x1, params)
        freqs, spect2 = current_expectation_power_spectrum(x2, params)
        if normalized:
            spect1 /= spect1.max()
            spect2 /= spect2.max()
        ax.semilogy(freqs / lat.freq, spect1, label="$U = {} \cdot t_0, a = {}$".format(x1[0] / params.t, x1[1]),
                    color="blue")
        ax.semilogy(freqs / lat.freq, spect2, label="$U = {} \cdot t_0, a = {}$".format(x2[0] / params.t, x2[1]),
                    color="red", linestyle="dashed")
        ax.set_xlabel("Harmonic Order")
        ax.set_ylabel("Power")
        # ax.set_xlim((0, 50))
        for i in range(1, 100, 2):
            ax.axvline(i, linestyle="dashed", color="grey")
        ax.legend(loc="upper right")
        plt.savefig(outfile + ".png")
        plt.show()
    else:
        raise Exception("Invalid specifier for domain")


target_U_over_t0 = 0
target_a = 4
sites = 6
sym = True

target_x, params = set_up(target_U_over_t0, target_a, sym, sites)

U_over_t0_min = 0
U_over_t0_max = 10
a_min = 1
a_max = 10

parameters = '-sites{}-U_min{}t0-U_max{}t0-{}a_min-{}a_max'.format(params.nx, U_over_t0_min, U_over_t0_max, a_min, a_max)
if sym:
    parameters += '-withsymmetry'
else:
    parameters += '-withoutsymmetry'

x_vals, bounds, U_vals, a_vals = get_data_points(params, U_over_t0_min, U_over_t0_max, a_min, a_max)

"""Load uncut spectra"""
spectra = np.load('./Spectra/spectra' + parameters + '.npy')

"""Get cost and plot it"""
# costs = get_cost(params, target_x, new_spectra, cut_indices)
# plot_cost(costs, parameters, params, target_x, U_vals, a_vals)

similarity_comparison(spectra, parameters)
# plot_response(params, [np.array([5*params.t, 5]), np.array([3*params.t, 4]), np.array([7*params.t, 8])], "frequency")
# compare_response(params, np.array([5*params.t, 5]), np.array([4*params.t, 9]), "time", normalized=True)
