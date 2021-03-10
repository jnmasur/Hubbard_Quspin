from minimization_methods import *
from tools import parameter_instantiate as hhg
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
from matplotlib import animation as ani



def set_up(symmetry, sites):
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

    # add all parameters to the class and create the basis
    params = Parameters(L, N_up, N_down, t0, field, F0, pbc)
    params.set_basis(symmetry)

    return params


def get_data_points(params, U_over_t0_lower=0, U_over_t0_upper=10, a_lower=1, a_upper=10):
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
    Saves spectra over a grid for processing, slices spectra from 2 omega_0 to 30 omega_0
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
            min_indx = np.searchsorted(freqs, 2)
            max_indx = np.searchsorted(freqs, 30)
            spect = spect[min_indx:max_indx]
            scaled_results.append(spect)
        scaled_results = np.array(scaled_results)
        np.save("./GridAnalysisData/Spectra" + parameters + ".npy", scaled_results)

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

        np.save("./GridAnalysisData/CurrentExpectations" + parameters + ".npy", results)
        return results


def similarity_comparison(arr, parameters, domain):
    if domain == "time":
        func = time_domain_objective
        outfile = "TimeDomainSimilarityData"
    elif domain == "frequency":
        func = objective_w_spectrum
        outfile = "SpectralSimilarityData"
    else:
        raise Exception("Invalid domain")

    if __name__ == "__main__":
        with Pool(100) as pool:
            similarity = [
                list(pool.map(func, ((x, y) for x in arr))) for y in arr
            ]
        similarity = np.array(similarity)
        np.save("./GridAnalysisData/" + outfile + parameters, similarity)
        return similarity


def graph_similarity(similarity, parameters, domain):
    if domain == "time":
        outfile = "TimeDomainSimilarityGraph"
    elif domain == "frequency":
        outfile = "SpectralSimilarityGraph"
    else:
        raise Exception("Invalid domain")
    plt.imshow(similarity, origin=True)
    plt.colorbar()
    plt.show()
    plt.savefig("./GridAnalysisData/" + outfile + parameters)


def animation_data(arr, parameters, domain):
    if domain == "time":
        func = time_domain_objective
        outfile = "TimeDomainAnimationData"
        arr = np.reshape(arr, (100, 100, 2000))
    elif domain == "frequency":
        func = objective_w_spectrum
        outfile = "SpectralAnimationData"
        arr = np.reshape(arr, (100, 100, 280))
    else:
        raise Exception("Invalid domain")

    if __name__ == "__main__":
        results = []
        with Pool(100) as pool:
            # U range is a list of either spectra or current expectations from (U_0->U_99) for a given a value
            for U_range in arr:
                print(U_range.shape)
                results.append([
                    list(pool.map(func, ((x, y) for x in U_range))) for y in U_range
                ])
        results = np.array(results)
        np.save("./GridAnalysisData/" + outfile + parameters, results)
        return results


def animation(ani_data, parameters, domain):
    """
    Animate a graph so that time runs with increasing a
    :param ani_data: an array whose ith entry is a grid with a range of U/t0 values
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if domain == "time":
        outfile = "TimeDomainAnimation"
        title_text = "Time Domain Similarity"
    elif domain == "frequency":
        outfile = "SpectralAnimation"
        title_text = "Frequency Domain Similarity"
    else:
        raise Exception("Invalid domain")

    writer = ani.PillowWriter()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    im = ax.imshow(ani_data[0], origin=True, extent=[0,10,0,10])
    bar = fig.colorbar(im, cax=cax)
    im.set_clim(np.max(ani_data), np.min(ani_data))
    ax.set_xlabel("$U/t_0$")
    ax.set_ylabel("$U/t_0$")
    title = ax.set_title(title_text + "at $a = 1.000$")

    def chart(i):
        im.set_data(ani_data[i])
        title.set_text(title_text + " at $a = {:.3f}$".format(1 + (1/11) * i))

    animator = ani.FuncAnimation(fig, chart, frames=100, repeat=False)
    plt.show()
    animator.save("./GridAnalysisData/" + outfile + parameters + ".gif", writer=writer)
    plt.close(fig)


sites = 6
sym = True

params = set_up(sym, sites)

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

"""Spectral Analysis"""
# spectra = get_spectra(params, x_vals, parameters)
# spectra = np.load('./GridAnalysisData/Spectra' + parameters + '.npy')
# similarity_comparison(spectra, parameters, "frequency")
# similarity = np.load('./GridAnalysisData/SpectralSimilarityData' + parameters + '.npy')
# graph_similarity(similarity, parameters, "frequency")
# ani_data = animation_data(spectra, parameters, "frequency")
ani_data = np.load("./GridAnalysisData/SpectralAnimationData" + parameters + ".npy")
animation(ani_data, parameters, "frequency")

"""Time Domain Analysis"""
# expecs = get_current_expecs(params, x_vals, parameters)
# expecs = np.load('./GridAnalysisData/CurrentExpectations' + parameters + '.npy')
# similarity_comparison(expecs, parameters, "time")
# similarity = np.load('./GridAnalysisData/TimeDomainSimilarityData' + parameters + '.npy')
# graph_similarity(similarity, parameters, "time")
# ani_data = animation_data(expecs, parameters, "time")
# ani_data = np.load("./GridAnalysisData/TimeDomainAnimationData" + parameters + ".npy")
# animation(ani_data, parameters, "time")

