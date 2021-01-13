from minimization_methods import *
from tools import parameter_instantiate as hhg
import numpy as np
from multiprocessing import cpu_count, Pool
from time import time
from warnings import warn
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
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
    ind = parameters.index('-sites')
    spectrum_parameters = parameters[ind:]

    if __name__ == "__main__":
        # this saves the spectra for processing
        with Pool(100) as pool:
            results = pool.starmap(current_expectation_power_spectrum, data_points)
        results = np.array(results)
        print(results.shape)

        new_results = []
        for pair in results:
            freqs, spect = pair
            freqs = freqs / lat.freq
            new_results.append((freqs, spect))
        new_results = np.array(new_results)
        print(new_results.shape)
        np.save("./Spectra/spectra" + spectrum_parameters + ".npy", new_results)

        return results


def get_cost(params, target_x, x_vals, parameters, spectra=None, zeroindex=None, cutdex=None):
    """
    Given a target_x and a grid (x_vals) find the cost for each point in the grid and save it
    :param spectra: an array of spectra
    :return: the costs over the grid for the fixed target
    """

    target_freqs, target_current = current_expectation_power_spectrum(target_x, params)

    if spectra is not None:
        if __name__ == "__main__":
            if zeroindex is not None and cutdex is not None:
                data_points = (((target_freqs[zeroindex:cutdex], target_current[zeroindex:cutdex]), res) for res in
                               spectra)
            else:
                data_points = (((target_freqs, target_current), res) for res in spectra)
            with Pool(100) as pool:
                costs = pool.starmap(objective_w_spectrum, data_points)
            costs = np.array(costs)
            np.save("./Data/CostData" + parameters + ".npy", costs)

            return costs

    else:
        data_points = ((np.array(point), target_current, params) for point in x_vals)
        if __name__ == "__main__":
            # this saves the costs for plotting
            with Pool(100) as pool:
                costs = pool.starmap(objective, data_points)
            costs = np.array(costs)
            np.save("./CostData/costs" + parameters + ".npy", costs)

            return costs


def plot_cost(costs, parameters, params, target_x, U_vals, a_vals, type='2d', add_title=False, parameters2=None):
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
        if parameters2 is not None:
            plt.savefig('./CostData/plot' + parameters + parameters2 + '.png')
        else:
            plt.savefig('./CostData/plot' + parameters + '.png')
        # plt.show()
        plt.close()

    # NOT FULLY IMPLEMENTED DONT USE THIS YET
    elif type == '3d':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(U_vals / params.t, a_vals, costs, cmap=cm.get_cmap('coolwarm'))
        plt.show()


def plot_spectra(params, first_x, second_x, outfile):
    """
    Plots two spectra on the same graph
    """
    # the lat is for getting the pulse frequency so it does not depend on U
    lat = hhg(params.field, params.nup, params.ndown, params.nx, 0, 0, params.t)

    first_freqs, first_spectrum = current_expectation_power_spectrum(first_x, params)
    second_freqs, second_spectrum = current_expectation_power_spectrum(second_x, params)

    first_freqs, second_freqs = first_freqs / lat.freq, second_freqs / lat.freq

    fig, ax = plt.subplots()
    ax.semilogy(first_freqs, first_spectrum, label="$U = {} \cdot t_0, a = {}$".format(first_x[0]/params.t, first_x[1]),
                color="blue")
    ax.semilogy(second_freqs, second_spectrum, label="$U = {} \cdot t_0, a = {}$".format(second_x[0]/params.t,
                                                                                         second_x[1]), color="orange")
    ax.set_xlabel("Harmonic Order")
    ax.set_ylabel("Power")
    ax.set_xlim((0, 50))
    for x in range(1, 50, 2):
        ax.axvline(x, linestyle="dashed", color="grey")
    ax.legend(loc="upper right")
    plt.savefig(outfile)
    plt.show()


def slice_spectra(spectra, cutoff, parameters):
    """
    Slices spectra at cutoff which is in terms of omega0
    :return: sliced_spectra, zeroindex, cutdex
    """

    ind = parameters.index('-sites')
    spectrum_parameters = parameters[ind:] + '-cutoff{}omega0'.format(cutoff)

    new_spectra = []
    zeroindex = 0
    cutdex = 0
    for pair in spectra:
        freqs, spect = pair
        cutdex = np.argmax(freqs > cutoff)
        spect = spect[:cutdex]
        freqs = freqs[:cutdex]
        # zeroindex = np.argmin(spect > 1e-20)
        zeroindex = find_zero(spect, 1e-20)
        spect = spect[zeroindex:]
        freqs = freqs[zeroindex:]
        new_spectra.append((freqs, spect))

    new_spectra = np.array(new_spectra)
    np.save('./Spectra/cutoffSpectra'+spectrum_parameters+'.npy', new_spectra)
    return new_spectra, zeroindex, cutdex


def find_zero(spectra, minimum):
    """
    Finds last occurrence less than minimum
    :param spectra: spectra we want to cut
    :param minimum: minimum value we will accept
    :return: the last occurrence less than minimum, if there are none, returns np.inf
    """
    zeroindex = np.inf
    for i in range(len(spectra)):
        if spectra[i] < minimum:
            zeroindex = i
    return zeroindex

target_U_over_t0 = 5
target_a = 5
sites = 6
sym = True

target_x, params = set_up(target_U_over_t0, target_a, sym, sites)

U_over_t0_min = 0
U_over_t0_max = 10
a_min = 0
a_max = 10

parameters = '-target_U{}t0-target_a{}-sites{}-U_min{}t0-U_max{}t0-{}a_min-{}a_max'.format(
            target_x[0] / params.t, target_x[1], params.nx, U_over_t0_min, U_over_t0_max, a_min, a_max)
if sym:
    parameters += '-withsymmetry'
else:
    parameters += '-withoutsymmetry'

x_vals, bounds, U_vals, a_vals = get_data_points(params, U_over_t0_min, U_over_t0_max, a_min, a_max)

# spectra = get_spectra(params, x_vals, parameters)

ind = parameters.index('-sites')
spectra_parameters = parameters[ind:]
spectra = np.load('./Spectra/spectra' + spectra_parameters+'.npy')

# for 10 -> 30 by 5s
for cutoff in range(10, 35, 5):
    sliced_spectrum, zeroindex, cutdex = slice_spectra(spectra, cutoff, parameters)
    costs = get_cost(params, target_x, x_vals, parameters, spectra=sliced_spectrum, zeroindex=zeroindex, cutdex=cutdex)
    plot_cost(costs, parameters, params, target_x, U_vals, a_vals, parameters2='-cutoff{}omega0'.format(cutoff))
