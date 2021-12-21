from minimization_methods import *
from tools import parameter_instantiate as hhg
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
from matplotlib import animation as ani
from matplotlib import colors


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

    # cycles of input field
    cycles = 10

    # add all parameters to the class and create the basis
    params = Parameters(L, N_up, N_down, t0, field, F0, pbc, cycles)
    params.set_basis(symmetry)

    return params


def get_data_points(params, U_over_t0_lower=0, U_over_t0_upper=10, a_lower=1, a_upper=10, random=False):
    """
    :returns: list of data points that form a grid over (U_lower, U_upper) * (a_lower, a_upper), bounds, U_vals, a_vals
    """

    U_lower = params.t * U_over_t0_lower
    U_upper = params.t * U_over_t0_upper

    bounds = ((U_lower, U_upper), (a_lower, a_upper))

    # randomness only used for animation, we don't need to sort U/t0
    if random:
        U_vals = (U_upper - U_lower) * np.random.random(99) + U_lower
        U_vals = np.insert(U_vals, 0, 0)
    else:
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
    data_points = ((np.array(point), params, True) for point in x_vals)

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


def cut_spectra(spectra, wmin, wmax, parameters):
    return


def get_spectra_wo_cuts(params, x_vals, parameters):
    """
    Saves spectra over a grid for processing
    :return: the spectra
    """
    data_points = ((np.array(point), params) for point in x_vals)

    if __name__ == "__main__":
        # this saves the spectra for processing
        with Pool(100) as pool:
            results = pool.starmap(current_expectation_power_spectrum, data_points)
        results = np.array(results)
        print(results.shape)

        np.save("./GridAnalysisData/Spectra" + parameters + ".npy", results)

        return results


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


def cost_comparison(arr, parameters, domain):
    if domain == "time":
        func = time_domain_objective
        outfile = "TimeDomainCostComparisonData"
    elif domain == "frequency":
        func = objective_w_spectrum
        outfile = "SpectralCostComparisonData"
    elif domain == "new time":
        func = new_objective
        outfile = "TimeDomainNewCostComparisonData"
    elif domain == "new frequency":
        func = new_objective
        outfile = "SpectralNewCostComparisonData"
    else:
        raise Exception("Invalid domain")

    if __name__ == "__main__":
        with Pool(100) as pool:
            cost_comparison = [
                list(pool.map(func, ((x, y) for x in arr))) for y in arr
            ]
        cost_comparison = np.array(cost_comparison)
        np.save("./GridAnalysisData/" + outfile + parameters, cost_comparison)
        return cost_comparison


def graph_cost_comparison(cost_comparison, parameters, domain):
    if domain == "time":
        outfile = "TimeDomainCostComparisonGraph"
    elif domain == "frequency":
        outfile = "SpectralCostComparisonGraph"
    else:
        raise Exception("Invalid domain")
    plt.imshow(cost_comparison, origin=True)
    plt.colorbar()
    plt.show()
    plt.savefig("./GridAnalysisData/" + outfile + parameters)


def animation_data(arr, parameters, domain):
    if domain == "time":
        func = time_domain_objective
        outfile = "TimeDomainAnimationData"
    elif domain == "frequency":
        func = objective_w_spectrum
        outfile = "SpectralAnimationData"
    elif domain == "new time":
        func = new_objective
        outfile = "NewTimeDomainAnimationData"
    elif domain == "new frequency":
        func = new_objective
        outfile = "NewSpectralAnimationData"
    else:
        raise Exception("Invalid domain")

    arr = np.reshape(arr, (100, 100, arr.size // (100 * 100)))

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
        title_text = "Time Domain Cost Comparison"
        mask_val = 1
    elif domain == "frequency":
        outfile = "SpectralAnimation"
        title_text = "Frequency Domain Cost Comparison"
        mask_val = 10
    else:
        raise Exception("Invalid domain")

    writer = ani.PillowWriter()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    masked_data = np.ma.array(ani_data, mask=ani_data < mask_val)
    masked_data /= masked_data.max()
    ax.set_xlabel("$U/t_0$")
    ax.set_ylabel("$U/t_0$")
    title = ax.set_title(title_text + "at $a = 1.000$")

    im = ax.imshow(masked_data[0], origin=True, extent=[0, 10, 0, 10], cmap="YlOrBr",
                   norm=colors.LogNorm(vmin=masked_data.min(), vmax=masked_data.max()))
    bar = fig.colorbar(im, cax=cax)
    im.set_clim(np.min(masked_data), np.max(masked_data))

    def chart(i):
        im.set_data(masked_data[i])
        title.set_text(title_text + " at $a = {:.3f}$".format(1 + (1 / 11) * i))

    animator = ani.FuncAnimation(fig, chart, frames=100, repeat=False)
    plt.show()
    animator.save("./GridAnalysisData/" + outfile + parameters + ".gif", writer=writer)
    plt.close(fig)


def scatter_plot(time_cost_comparison, frequency_cost_comparison, parameters):
    """
    x axis is time domain cost and y axis is frequency cost, both are normalized
    """
    time_cost_comparison = time_cost_comparison.flatten()
    frequency_cost_comparison = frequency_cost_comparison.flatten()
    # normalize both on a scale from 0 to 1
    time_cost_comparison /= time_cost_comparison.max()
    frequency_cost_comparison /= frequency_cost_comparison.max()
    plt.scatter(time_cost_comparison, frequency_cost_comparison)
    plt.xlabel("Time domain cost")
    plt.ylabel("Frequency domain cost")
    plt.show()


def scatter_plot_animation(time_ani, frequency_ani, parameters):
    """
    If randomly generated data is passed in instead of a grid, there
    will be random coloring of the points
    """

    writer = ani.PillowWriter()
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    ax = fig.add_subplot(111)
    ax.set_xlabel("Current Density Cost")
    ax.set_ylabel("Power Spectrum cost")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # THIS DIFFERENCE WILL NOT WORK WITH RANDOM POINTS, OR ANY U BOUNDS BESIDES (0, 10*t_0)
    U_range, delta_U = np.linspace(0, 5.2, num=100, retstep=True)
    difference = np.array([[x - y for x in U_range] for y in U_range])
    # remove data duplicates by taking upper triangle and removing 0s
    time_data = np.array([np.triu(time_ani[i]) for i in range(len(time_ani))])
    frequency_data = np.array([np.triu(frequency_ani[i]) for i in range(len(frequency_ani))])
    difference = np.triu(difference)
    inds = np.nonzero(time_data)
    diff_inds = np.nonzero(difference)
    time_data = time_data[inds]
    frequency_data = frequency_data[inds]
    difference = difference[diff_inds]
    n = len(time_data) // 100
    time_data = np.array([time_data[i * n:(i + 1) * n] for i in range(100)])
    frequency_data = np.array([frequency_data[i * n:(i + 1) * n] for i in range(100)])
    # separate conductors from insulators
    # cond_time_data = np.array([time_data[j, :99] / time_data[j].max() for j in range(100)])
    # cond_frequency_data = np.array([frequency_data[j, :99] / frequency_data[j].max() for j in range(100)])
    # ins_time_data = np.array([time_data[j, 99:] / time_data[j].max() for j in range(100)])
    # ins_frequency_data = np.array([frequency_data[j, 99:] / frequency_data[j].max() for j in range(100)])
    cond_time_data = np.array([time_data[j, :99] / time_data.max() for j in range(100)])
    cond_frequency_data = np.array([frequency_data[j, :99] / frequency_data.max() for j in range(100)])
    ins_time_data = np.array([time_data[j, 99:] / time_data.max() for j in range(100)])
    ins_frequency_data = np.array([frequency_data[j, 99:] / frequency_data.max() for j in range(100)])

    ins_difference = difference[99:]
    cmp = np.linspace(0, 1, num=99)  # there are 99 unique values for comparisons: U_1 - U_0, ... , U_99 - U_0
    imp = np.linspace(0, 1, num=98)  # there are 98 unique values for comparisons: U_2 - U_1, ... , U_99 - U_1
    cond_diff_vals = U_range[1:]
    ins_diff_vals = U_range[1:-1]
    cond_sctrs = []
    ins_sctrs = []
    # the lists below have the same length as ins_sctr (98), but each element is a list itself of all comparisons
    # that have the same difference
    ins_time_sorted = []
    ins_frequency_sorted = []

    def initializer():
        for i in range(len(cond_diff_vals)):
            color = plt.cm.Reds(cmp[i])
            cond_sctrs.append(ax.plot(cond_time_data[0][i], cond_frequency_data[0][i], ls='', color=color,
                                      marker='.')[0])

        for i in range(len(ins_diff_vals)):
            color = plt.cm.Blues(imp[i])
            point_inds = np.where(ins_difference == ins_diff_vals[i])[0]
            temp_time = []
            temp_frequency = []
            for t in range(100):
                time_t = []
                freq_t = []
                for point_ind in point_inds:
                    time_t.append(ins_time_data[t][point_ind])
                    freq_t.append(ins_frequency_data[t][point_ind])
                temp_time.append(time_t)
                temp_frequency.append(freq_t)
            ins_time_sorted.append(temp_time)
            ins_frequency_sorted.append(temp_frequency)
            ins_sctrs.append(ax.plot(temp_time[0], temp_frequency[0], ls='', color=color, marker='.')[0])

    def chart(frame):
        ax.set_title("Comparison of Current Density and Power Spectrum Costs $a = {:.3f}$".format(1 + (1 / 11) * frame))
        for j in range(len(cond_sctrs)):
            cond_sctrs[j].set_data(cond_time_data[frame][j], cond_frequency_data[frame][j])
        for j in range(len(ins_sctrs)):
            ins_sctrs[j].set_data(ins_time_sorted[j][frame], ins_frequency_sorted[j][frame])

    initializer()
    for i in range(100):
        chart(i)
        plt.savefig("./AnimationImages/p{}.png".format(i+1))
    # chart(99)
    # plt.show()

    # animator = ani.FuncAnimation(fig, chart, init_func=initializer, frames=100, repeat=False)
    # plt.show()
    # animator.save("./GridAnalysisData/ScatterAnimation" + parameters + ".gif", writer=writer)
    plt.close(fig)


def paper_graph(time_ani, frequency_ani):
    """
    If randomly generated data is passed in instead of a grid, there
    will be random coloring of the points
    """
    fig = plt.figure()
    fig.set_figheight(9)
    fig.set_figwidth(4)
    ax1, ax2, ax3 = fig.subplots(3, sharex=True, sharey=True)
    ax3.set_xlabel("Current Density Distance $(\\mathcal{D}_t)$")
    ax2.set_ylabel("Power Spectrum Distance $(\\mathcal{D}_p)$")
    # fig.set_figheight(4)
    # fig.set_figwidth(12)
    # ax1, ax2, ax3 = fig.subplots(1, 3, sharex=True, sharey=True)
    # ax2.set_xlabel("Current Density Cost $(C_t)$")
    # ax1.set_ylabel("Power Spectrum Cost $(C_p)$")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    plt.text(.91, 3.31, '(a)', fontsize='x-large')
    plt.text(.91, 2.11, '(b)', fontsize='x-large')
    plt.text(.91, .91, '(c)', fontsize='x-large')
    # plt.text(-2.4, .93, '(a)', fontsize='x-large')
    # plt.text(-1.2, .93, '(b)', fontsize='x-large')
    # plt.text(0, .93, '(c)', fontsize='x-large')
    # plt.text(-2.4, 240, '(a)', fontsize='x-large')
    # plt.text(-1.2, 240, '(b)', fontsize='x-large')
    # plt.text(0, 240, '(c)', fontsize='x-large')

    # THIS DIFFERENCE WILL NOT WORK WITH RANDOM POINTS, OR ANY U BOUNDS BESIDES (0, 10*t_0)
    U_range, delta_U = np.linspace(0, 5.2, num=100, retstep=True)
    # remove data duplicates by taking upper triangle and removing 0s
    time_data = np.array([np.triu(time_ani[i], k=1) for i in range(len(time_ani))])
    frequency_data = np.array([np.triu(frequency_ani[i], k=1) for i in range(len(frequency_ani))])
    inds = np.nonzero(time_data)
    time_data = time_data[inds]
    frequency_data = frequency_data[inds]
    n = len(time_data) // 100
    time_data = np.array([time_data[i * n:(i + 1) * n] for i in range(100)])
    frequency_data = np.array([frequency_data[i * n:(i + 1) * n] for i in range(100)])
    time_max = time_data.max()
    frequency_max = frequency_data.max()
    # separate conductors from insulators
    # cond_time_data = np.array([time_data[j, :99] / time_data[j].max() for j in range(100)])
    # cond_frequency_data = np.array([frequency_data[j, :99] / frequency_data[j].max() for j in range(100)])
    # ins_time_data = np.array([time_data[j, 99:] / time_data[j].max() for j in range(100)])
    # ins_frequency_data = np.array([frequency_data[j, 99:] / frequency_data[j].max() for j in range(100)])
    cond_time_data = np.array([time_data[j, :99] for j in range(100)]) / time_max
    cond_frequency_data = np.array([frequency_data[j, :99] for j in range(100)]) / frequency_max
    ins_time_data = np.array([time_data[j, 99:] for j in range(100)]) / time_max
    ins_frequency_data = np.array([frequency_data[j, 99:] for j in range(100)]) / frequency_max

    coloring = "average"
    # coloring = "difference"
    cmp = np.linspace(0, 1, num=99)  # there are 99 unique averages and differences

    if coloring == "average":
        average = np.array([[(x + y) / 2 for x in U_range] for y in U_range])
        average = np.triu(average, k=1)
        avg_inds = np.nonzero(average)
        average = average[avg_inds]
        ins_average = average[99:]
        imp_avg = np.linspace(0, 1, num=195)  # there are 195 unique averages
        cond_avg_vals = U_range[1:] / 2
        ins_avg_vals = np.linspace(1.5 * delta_U, 197 * delta_U / 2, num=195)
        ins_comparisons = ins_average
        cond_color_vals = cond_avg_vals
        ins_color_vals = ins_avg_vals
        imp = imp_avg
    else:
        difference = np.array([[x - y for x in U_range] for y in U_range])
        difference = np.triu(difference, k=1)
        diff_inds = np.nonzero(difference)
        difference = difference[diff_inds]
        ins_difference = difference[99:]
        imp_diff = np.linspace(0, 1, num=98)
        cond_diff_vals = U_range[1:]
        ins_diff_vals = U_range[1:-1]
        ins_comparisons = ins_difference
        cond_color_vals = cond_diff_vals
        ins_color_vals = ins_diff_vals
        imp = imp_diff

    ins_time_sorted = []
    ins_frequency_sorted = []

    for i in range(len(cond_color_vals)):
        color = plt.cm.Reds(cmp[i])
        _ = ax1.plot(cond_time_data[0, i], cond_frequency_data[0, i], ls='', color=color, marker='.')
        _ = ax2.plot(cond_time_data[5, i], cond_frequency_data[5, i], ls='', color=color, marker='.')
        _ = ax3.plot(cond_time_data[99, i], cond_frequency_data[99, i], ls='', color=color, marker='.')

    print(len(ins_comparisons))
    tot = 0
    for i in range(len(ins_color_vals)):
        color = plt.cm.Blues(imp[i])
        point_inds = np.where(abs(ins_comparisons - ins_color_vals[i]) < .01)[0]
        tot += point_inds.shape[0]
        temp_time = []
        temp_frequency = []
        for t in range(100):
            time_t = []
            freq_t = []
            for point_ind in point_inds:
                time_t.append(ins_time_data[t][point_ind])
                freq_t.append(ins_frequency_data[t][point_ind])
            temp_time.append(time_t)
            temp_frequency.append(freq_t)
        ins_time_sorted.append(temp_time)
        ins_frequency_sorted.append(temp_frequency)
        ax1.plot(temp_time[0], temp_frequency[0], ls='', color=color, marker='.')
        ax2.plot(temp_time[5], temp_frequency[5], ls='', color=color, marker='.')
        ax3.plot(temp_time[99], temp_frequency[99], ls='', color=color, marker='.')

    # add x = y line to each plot
    ys = xs = np.linspace(0, 1, num=100)
    ax1.plot(xs, ys, color="black", ls="dashed")
    ax2.plot(xs, ys, color="black", ls="dashed")
    ax3.plot(xs, ys, color="black", ls="dashed")

    plt.show()


def paper_graph_F0(time_ani, frequency_ani):
    """
    If randomly generated data is passed in instead of a grid, there
    will be random coloring of the points
    """
    fig = plt.figure()
    # fig.set_figheight(9)
    # fig.set_figwidth(4)
    # ax1, ax2, ax3 = fig.subplots(3, sharex=True, sharey=True)
    # ax3.set_xlabel("Current Density Cost $(C_t)$")
    # ax2.set_ylabel("Power Spectrum Cost $(C_p)$")
    fig.set_figheight(4)
    fig.set_figwidth(12)
    ax1, ax2, ax3 = fig.subplots(1, 3, sharex=True, sharey=True)
    ax2.set_xlabel("Current Density Cost $(C_t)$")
    ax1.set_ylabel("Power Spectrum Cost $(C_p)$")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    plt.text(-2.4, .93, '(a)', fontsize='x-large')
    plt.text(-1.2, .93, '(b)', fontsize='x-large')
    plt.text(0, .93, '(c)', fontsize='x-large')

    curr1 = time_ani[0]
    spec1 = frequency_ani[0]

    curr2 = np.load("./GridAnalysisData/TimeDomainCostComparisonData-sites10-U_min0t0-U_max10t0-1a_min-10a_max-withsymmetry-28.900000000000002F0.npy")
    spec2 = np.load("./GridAnalysisData/SpectralCostComparisonData-sites10-U_min0t0-U_max10t0-1a_min-10a_max-withsymmetry-28.900000000000002F0.npy")

    curr3 = np.load("./GridAnalysisData/TimeDomainCostComparisonData-sites10-U_min0t0-U_max10t0-1a_min-10a_max-withsymmetry-100.0F0.npy")
    spec3 = np.load("./GridAnalysisData/SpectralCostComparisonData-sites10-U_min0t0-U_max10t0-1a_min-10a_max-withsymmetry-100.0F0.npy")

    # THIS DIFFERENCE WILL NOT WORK WITH RANDOM POINTS, OR ANY U BOUNDS BESIDES (0, 10*t_0)
    U_range, delta_U = np.linspace(0, 5.2, num=100, retstep=True)
    difference = np.array([[x - y for x in U_range] for y in U_range])
    # remove data duplicates by taking upper triangle and removing 0s
    curr1 = curr1[np.nonzero(np.triu(curr1))] / curr1.max()
    spec1 = spec1[np.nonzero(np.triu(spec1))] / spec1.max()
    curr2 = curr2[np.nonzero(np.triu(curr2))] / curr2.max()
    spec2 = spec2[np.nonzero(np.triu(spec2))] / spec2.max()
    curr3 = curr3[np.nonzero(np.triu(curr3))] / curr3.max()
    spec3 = spec3[np.nonzero(np.triu(spec3))] / spec3.max()
    difference = difference[np.nonzero(np.triu(difference))]

    cond_curr1 = curr1[:99]
    cond_spec1 = spec1[:99]
    cond_curr2 = curr2[:99]
    cond_spec2 = spec2[:99]
    cond_curr3 = curr3[:99]
    cond_spec3 = spec3[:99]
    ins_curr1 = curr1[99:]
    ins_spec1 = spec1[99:]
    ins_curr2 = curr2[99:]
    ins_spec2 = spec2[99:]
    ins_curr3 = curr3[99:]
    ins_spec3 = spec3[99:]
    ins_difference = difference[99:]
    cmp = np.linspace(0, 1, num=99)  # there are 99 unique values for comparisons: U_1 - U_0, ... , U_99 - U_0
    imp = np.linspace(0, 1, num=98)  # there are 98 unique values for comparisons: U_2 - U_1, ... , U_99 - U_1
    cond_diff_vals = U_range[1:]
    ins_diff_vals = U_range[1:-1]

    for i in range(len(cond_diff_vals)):
        color = plt.cm.Reds(cmp[i])
        ax1.plot(cond_curr1[i], cond_spec1[i], ls='', color=color, marker='.')
        ax2.plot(cond_curr2[i], cond_spec2[i], ls='', color=color, marker='.')
        ax3.plot(cond_curr3[i], cond_spec3[i], ls='', color=color, marker='.')

    tot = 0
    print(len(ins_difference))
    for i in range(len(ins_diff_vals)):
        color = plt.cm.Blues(imp[i])
        point_inds = np.where(abs(ins_difference - ins_diff_vals[i]) < .01)[0]
        tot += point_inds.shape[0]
        ax1.plot(ins_curr1[point_inds], ins_spec1[point_inds], ls='', color=color, marker='.')
        ax2.plot(ins_curr2[point_inds], ins_spec2[point_inds], ls='', color=color, marker='.')
        ax3.plot(ins_curr3[point_inds], ins_spec3[point_inds], ls='', color=color, marker='.')

    print(tot)
    plt.show()


def scatter_plot_field(params, parameters):
    U_vals = np.linspace(0, 10 * params.t, num=100)
    a = 1
    x_vals = [(U, a) for U in U_vals]

    delta_F = (100 - 10) / 100

    F1 = params.F0
    print(F1)
    parameters1 = parameters + "-{}F0".format(F1)
    currents1 = get_current_expecs(params, x_vals, parameters1)
    spectra1 = get_spectra(params, x_vals, parameters1)

    params.F0 = 10 + delta_F * 21
    F2 = params.F0
    print(F2)
    parameters2 = parameters + "-{}F0".format(F2)
    currents2 = get_current_expecs(params, x_vals, parameters2)
    spectra2 = get_spectra(params, x_vals, parameters2)

    params.F0 = 10 + delta_F * 100
    F3 = params.F0
    print(F3)
    parameters3 = parameters + "-{}F0".format(F3)
    currents3 = get_current_expecs(params, x_vals, parameters3)
    spectra3 = get_spectra(params, x_vals, parameters3)
    
    # params.F0 = 1000
    # F4 = params.F0
    # print(F4)
    # parameters4 = parameters + "-{}F0".format(F4)
    # currents4 = get_current_expecs(params, x_vals, parameters4)
    # spectra4 = get_spectra(params, x_vals, parameters4)
    
    time_cost1 = cost_comparison(currents1, parameters1, "time")
    freq_cost1 = cost_comparison(spectra1, parameters1, "frequency")

    time_cost2 = cost_comparison(currents2, parameters2, "time")
    freq_cost2 = cost_comparison(spectra2, parameters2, "frequency")

    time_cost3 = cost_comparison(currents3, parameters3, "time")
    freq_cost3 = cost_comparison(spectra3, parameters3, "frequency")

    # time_cost4 = cost_comparison(currents4, parameters4, "time")
    # freq_cost4 = cost_comparison(spectra4, parameters4, "frequency")

    time_cost1 = np.load("./GridAnalysisData/TimeDomainCostComparisonData" + parameters1 + ".npy")
    freq_cost1 = np.load("./GridAnalysisData/SpectralCostComparisonData" + parameters1 + ".npy")

    time_cost2 = np.load("./GridAnalysisData/TimeDomainCostComparisonData" + parameters2 + ".npy")
    freq_cost2 = np.load("./GridAnalysisData/SpectralCostComparisonData" + parameters2 + ".npy")

    time_cost3 = np.load("./GridAnalysisData/TimeDomainCostComparisonData" + parameters3 + ".npy")
    freq_cost3 = np.load("./GridAnalysisData/SpectralCostComparisonData" + parameters3 + ".npy")

    # time_cost4 = np.load("./GridAnalysisData/TimeDomainCostComparisonData" + parameters4 + ".npy")
    # freq_cost4 = np.load("./GridAnalysisData/SpectralCostComparisonData" + parameters4 + ".npy")

    time_cost1 = time_cost1.flatten() / time_cost1.max()
    freq_cost1 = freq_cost1.flatten() / freq_cost1.max()

    time_cost2 = time_cost2.flatten() / time_cost2.max()
    freq_cost2 = freq_cost2.flatten() / freq_cost2.max()

    time_cost3 = time_cost3.flatten() / time_cost3.max()
    freq_cost3 = freq_cost3.flatten() / freq_cost3.max()

    # time_cost4 = time_cost4.flatten() / time_cost4.max()
    # freq_cost4 = freq_cost4.flatten() / freq_cost4.max()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time Domain Cost $(C_t)$")
    ax.set_ylabel("Frequency Domain Cost $(C_p)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ax.set_title("Comparison at F0 = {}".format(F1))
    # ax.plot(time_cost1, freq_cost1, '.b')
    # plt.show()

    # ax.set_title("Comparison at F0 = {}".format(F2))
    # ax.plot(time_cost2, freq_cost2, '.b')
    # plt.show()

    # ax.set_title("Comparison at F0 = {}".format(F3))
    # ax.plot(time_cost3, freq_cost3, '.b')
    # plt.show()

    # ax.set_title("Comparison at F0 = {}".format(F4))
    # ax.plot(time_cost4, freq_cost4, '.b')
    # plt.show()


def scatter_plot_cycles(params, parameters):
    U_vals = np.linspace(0, 10 * params.t, num=100)
    a = 1
    x_vals = [(U, a) for U in U_vals]

    cycles1 = 10
    cycles2 = 20
    cycles3 = 30

    params.cycles = cycles1
    parameters1 = parameters + "-{}cycles".format(cycles1)
    # currents1 = get_current_expecs(params, x_vals, parameters1)
    # spectra1 = get_spectra(params, x_vals, parameters1)

    params.cycles = cycles2
    parameters2 = parameters + "-{}cycles".format(cycles2)
    # currents2 = get_current_expecs(params, x_vals, parameters2)
    # spectra2 = get_spectra(params, x_vals, parameters2)

    params.cycles = cycles3
    parameters3 = parameters + "-{}cycles".format(cycles3)
    # currents3 = get_current_expecs(params, x_vals, parameters3)
    # spectra3 = get_spectra(params, x_vals, parameters3)

    # time_cost1 = cost_comparison(currents1, parameters1, "time")
    # freq_cost1 = cost_comparison(spectra1, parameters1, "frequency")

    # time_cost2 = cost_comparison(currents2, parameters2, "time")
    # freq_cost2 = cost_comparison(spectra2, parameters2, "frequency")

    # time_cost3 = cost_comparison(currents3, parameters3, "time")
    # freq_cost3 = cost_comparison(spectra3, parameters3, "frequency")

    time_cost1 = np.load("./GridAnalysisData/TimeDomainCostComparisonData" + parameters1 + ".npy")
    freq_cost1 = np.load("./GridAnalysisData/SpectralCostComparisonData" + parameters1 + ".npy")

    time_cost2 = np.load("./GridAnalysisData/TimeDomainCostComparisonData" + parameters2 + ".npy")
    freq_cost2 = np.load("./GridAnalysisData/SpectralCostComparisonData" + parameters2 + ".npy")

    time_cost3 = np.load("./GridAnalysisData/TimeDomainCostComparisonData" + parameters3 + ".npy")
    freq_cost3 = np.load("./GridAnalysisData/SpectralCostComparisonData" + parameters3 + ".npy")

    time_cost1 = time_cost1.flatten() / time_cost1.max()
    freq_cost1 = freq_cost1.flatten() / freq_cost1.max()

    time_cost2 = time_cost2.flatten() / time_cost2.max()
    freq_cost2 = freq_cost2.flatten() / freq_cost2.max()

    time_cost3 = time_cost3.flatten() / time_cost3.max()
    freq_cost3 = freq_cost3.flatten() / freq_cost3.max()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time Domain Cost $(C_t)$")
    ax.set_ylabel("Frequency Domain Cost $(C_p)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title("Comparison at cycles = {}".format(cycles1))
    ax.plot(time_cost1, freq_cost1, '.b')
    plt.show()

    # ax.set_title("Comparison at cycles = {}".format(cycles2))
    # ax.plot(time_cost2, freq_cost2, '.b')
    # plt.show()

    # ax.set_title("Comparison at cycles = {}".format(cycles3))
    # ax.plot(time_cost3, freq_cost3, '.b')
    # plt.show()


def scatter_plot_animation_new(time_ani, frequency_ani, parameters):
    cut_time = [_.flatten()[:110] / _.max() for _ in time_ani]
    cut_frequency = [_.flatten()[:110] / _.max() for _ in frequency_ani]

    writer = ani.PillowWriter()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time domain cost")
    ax.set_ylabel("Frequency domain cost")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    sctr, = ax.plot(cut_time[0], cut_frequency[0], '.b')

    def chart(i):
        # time_data = (time_ani[i]) / (time_ani[i].max()).flatten()
        # frequency_data = (frequency_ani[i] / frequency_ani[i].max()).flatten()
        sctr.set_data(cut_time[i], cut_frequency[i])
        ax.set_title("Comparision of frequency and time domain costs $a = {:.3f}$".format(1 + (1 / 11) * i))

    animator = ani.FuncAnimation(fig, chart, frames=100, repeat=False)
    plt.show()
    animator.save("./GridAnalysisData/ScatterAnimationExperiment" + parameters + ".gif", writer=writer)
    plt.close(fig)


def current_animation(U_over_t0, parameters, params):
    a_values = np.linspace(1, 10, 50)
    x_ax = current_expectation((U_over_t0 * params.t, 1), params, True)[0]
    try:
        data = np.load("./GridAnalysisData/CurrentDensityExpectationAnimationData-U_over_t{}".format(
            U_over_t0) + parameters + ".npy")
    except:
        data = np.array([current_expectation((U_over_t0 * params.t, a), params) for a in a_values])
        np.save("./GridAnalysisData/CurrentDensityExpectationAnimationData-U_over_t{}".format(
            U_over_t0) + parameters + ".npy", data)

    writer = ani.PillowWriter()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time")
    ax.set_ylabel("Current Density")
    plot, = ax.plot(x_ax, data[0])
    ax.set_ylim(data.min(), data.max())

    def chart(i):
        plot.set_ydata(data[i])
        ax.set_title(
            "Current Density Expectation" + " at $U = {}\\cdot t_0, a = {:.3f}$".format(U_over_t0, 1 + (9 / 49) * i))

    animator = ani.FuncAnimation(fig, chart, frames=50, repeat=False)
    plt.show()
    animator.save(
        "./GridAnalysisData/CurrentDensityExpectationAnimation-U_over_t{}".format(U_over_t0) + parameters + ".gif",
        writer=writer)
    plt.close(fig)


def spectrum_animation(U_over_t0, parameters, params):
    lat = hhg(params.field, params.nup, params.ndown, params.nx, 0, 0, params.t)

    a_values = np.linspace(1, 10, 50)
    x_ax = current_expectation_power_spectrum((U_over_t0 * params.t, 1), params, True)[0] / lat.freq
    try:
        data = np.load(
            "./GridAnalysisData/CurrentSpectrumAnimationData-U_over_t{}".format(U_over_t0) + parameters + ".npy")
    except:
        data = np.array([current_expectation_power_spectrum((U_over_t0 * params.t, a), params) for a in a_values])
        np.save("./GridAnalysisData/CurrentSpectrumAnimationData-U_over_t{}".format(U_over_t0) + parameters + ".npy",
                data)

    writer = ani.PillowWriter()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Harmonic Order")
    ax.set_ylabel("Power")
    plot, = ax.semilogy(x_ax, data[0])
    ax.set_xlim(0, 75)
    ax.set_ylim(data.min(), data.max())

    def chart(i):
        plot.set_ydata(data[i])
        # ax.set_ylim(data[i].min(), data[i].max())
        ax.set_title("Current Power Spectrum at $U = {}\\cdot t_0, a = {:.3f}$".format(U_over_t0, 1 + (9 / 49) * i))

    animator = ani.FuncAnimation(fig, chart, frames=50, repeat=False)
    plt.show()
    # animator.save("./GridAnalysisData/CurrentSpectrumAnimation-U_over_t{}".format(U_over_t0) + parameters + ".gif", writer=writer)
    plt.close(fig)


def multiple_current_animation(U_over_t0s, parameters, params):
    """This is set up to only work with 3 values of U/t0"""

    a_values = np.linspace(1, 10, 50)
    x_ax = current_expectation((U_over_t0s[0] * params.t, 1), params, True)[0]

    data = []
    for U_over_t0 in U_over_t0s:
        try:
            d = np.load("./GridAnalysisData/CurrentDensityExpectationAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy")
        except:
            d = np.array([current_expectation((U_over_t0 * params.t, a), params) for a in a_values])
            np.save("./GridAnalysisData/CurrentDensityExpectationAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy", d)
        data.append(d)

    spectrum_data = []
    for U_over_t0 in U_over_t0s:
        try:
            d = np.load("./GridAnalysisData/CurrentSpectrumAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy")
        except:
            d = np.array([current_expectation_power_spectrum((U_over_t0 * params.t, a), params) for a in a_values])
            np.save("./GridAnalysisData/CurrentSpectrumAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy", d)
        spectrum_data.append(d)

    scatterx = []
    scattery = []
    labels = []
    for i in range(len(a_values)):
        temp_time = []
        temp_freq = []
        for j in range(len(U_over_t0s)):
            for k in range(len(U_over_t0s)):
                if j < k:
                    temp_time.append(time_domain_objective((data[j][i], data[k][i])))
                    temp_freq.append(objective_w_spectrum((spectrum_data[j][i], spectrum_data[k][i])))
                    if len(labels) < 3:
                        labels.append("$({} \\cdot t_0, {} \\cdot t_0)$".format(U_over_t0s[j], U_over_t0s[k]))
        temp_time = np.array(temp_time)
        temp_freq = np.array(temp_freq)
        scatterx.append(temp_time)
        scattery.append(temp_freq)
    scatterx = np.array(scatterx)
    scatterx /= scatterx.max()
    scattery = np.array(scattery)
    scattery /= scattery.max()

    data1, data2, data3 = data
    U_over_t01, U_over_t02, U_over_t03 = U_over_t0s
    writer = ani.PillowWriter()
    fig, axs = plt.subplots(2)
    fig.set_figheight(9)
    fig.set_figwidth(12)
    title = plt.suptitle("$a = 0.000$")

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Current Density")
    axs[0].set_title("Current Density Expectation")
    plot1, plot2, plot3 = axs[0].plot(x_ax, data1[0], "r", x_ax, data2[0], "b", x_ax, data3[0], "g")
    plot1.set_label("$U = {:.2f} \\cdot t_0$".format(U_over_t01))
    plot2.set_label("$U = {:.2f} \\cdot t_0$".format(U_over_t02))
    plot3.set_label("$U = {:.2f} \\cdot t_0$".format(U_over_t03))
    axs[0].legend(loc="upper right")

    axs[1].set_xlabel("Time Domain Cost")
    axs[1].set_ylabel("Frequency Domain Cost")
    sctr1, = axs[1].plot(scatterx[0][0], scattery[0][0], '.b', label=labels[0])
    sctr2, = axs[1].plot(scatterx[0][1], scattery[0][1], '.r', label=labels[1])
    sctr3, = axs[1].plot(scatterx[0][2], scattery[0][2], '.g', label=labels[2])
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend(loc="upper left")

    def chart(i):
        title.set_text("$a = {:.3f}$".format(1 + (9 / 49) * i))
        plot1.set_ydata(data1[i])
        plot2.set_ydata(data2[i])
        plot3.set_ydata(data3[i])
        sctr1.set_xdata(scatterx[i][0])
        sctr1.set_ydata(scattery[i][0])
        sctr2.set_xdata(scatterx[i][1])
        sctr2.set_ydata(scattery[i][1])
        sctr3.set_xdata(scatterx[i][2])
        sctr3.set_ydata(scattery[i][2])

    animator = ani.FuncAnimation(fig, chart, frames=50, repeat=False)
    plt.show()
    animator.save(
        "./GridAnalysisData/MultipleCurrentAnimation-U_over_t{}-{}-{}".format(U_over_t01, U_over_t02, U_over_t03)
        + parameters + ".gif", writer=writer)
    plt.close(fig)


def multiple_spectrum_animation(U_over_t0s, parameters, params):
    """This is set up to only work with 3 values of U/t0"""

    lat = hhg(params.field, params.nup, params.ndown, params.nx, 0, 0, params.t)

    a_values = np.linspace(1, 10, 50)
    x_ax = current_expectation_power_spectrum((U_over_t0s[0] * params.t, 1), params, True)[0] / lat.freq

    data = []
    for U_over_t0 in U_over_t0s:
        try:
            d = np.load("./GridAnalysisData/CurrentSpectrumAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy")
        except:
            d = np.array([current_expectation_power_spectrum((U_over_t0 * params.t, a), params) for a in a_values])
            np.save("./GridAnalysisData/CurrentSpectrumAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy", d)
        data.append(d)

    current_data = []
    for U_over_t0 in U_over_t0s:
        try:
            d = np.load("./GridAnalysisData/CurrentDensityExpectationAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy")
        except:
            d = np.array([current_expectation((U_over_t0 * params.t, a), params) for a in a_values])
            np.save("./GridAnalysisData/CurrentDensityExpectationAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy", d)
        current_data.append(d)

    scatterx = []
    scattery = []
    labels = []
    for i in range(len(a_values)):
        temp_time = []
        temp_freq = []
        for j in range(len(U_over_t0s)):
            for k in range(len(U_over_t0s)):
                if j < k:
                    temp_time.append(time_domain_objective((current_data[j][i], current_data[k][i])))
                    temp_freq.append(objective_w_spectrum((data[j][i], data[k][i])))
                    if len(labels) < 3:
                        labels.append("$({} \\cdot t_0, {} \\cdot t_0)$".format(U_over_t0s[j], U_over_t0s[k]))
        temp_time = np.array(temp_time)
        temp_freq = np.array(temp_freq)
        scatterx.append(temp_time)
        scattery.append(temp_freq)
    scatterx = np.array(scatterx)
    scatterx /= scatterx.max()
    scattery = np.array(scattery)
    scattery /= scattery.max()

    data1, data2, data3 = data
    U_over_t01, U_over_t02, U_over_t03 = U_over_t0s
    writer = ani.PillowWriter()
    fig, axs = plt.subplots(2)
    fig.set_figheight(9)
    fig.set_figwidth(12)
    title = plt.suptitle("$a = 0.000$")

    axs[0].set_xlabel("Harmonic Order")
    axs[0].set_ylabel("Power")
    axs[0].set_title("Current Power Spectrum")
    plot1, plot2, plot3 = axs[0].semilogy(x_ax, data1[0], "r", x_ax, data2[0], "b", x_ax, data3[0], "g")
    axs[0].set_xlim(0, 55)
    plot1.set_label("$U = {:.2f} \\cdot t_0$".format(U_over_t01))
    plot2.set_label("$U = {:.2f} \\cdot t_0$".format(U_over_t02))
    plot3.set_label("$U = {:.2f} \\cdot t_0$".format(U_over_t03))
    axs[0].legend(loc="upper right")

    axs[1].set_xlabel("Time Domain Cost")
    axs[1].set_ylabel("Frequency Domain Cost")
    sctr1, = axs[1].plot(scatterx[0][0], scattery[0][0], '.b', label=labels[0])
    sctr2, = axs[1].plot(scatterx[0][1], scattery[0][1], '.r', label=labels[1])
    sctr3, = axs[1].plot(scatterx[0][2], scattery[0][2], '.g', label=labels[2])
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend(loc="upper left")

    def chart(i):
        title.set_text("$a = {:.3f}$".format(1 + (9 / 49) * i))
        plot1.set_ydata(data1[i])
        plot2.set_ydata(data2[i])
        plot3.set_ydata(data3[i])
        sctr1.set_xdata(scatterx[i][0])
        sctr1.set_ydata(scattery[i][0])
        sctr2.set_xdata(scatterx[i][1])
        sctr2.set_ydata(scattery[i][1])
        sctr3.set_xdata(scatterx[i][2])
        sctr3.set_ydata(scattery[i][2])

    animator = ani.FuncAnimation(fig, chart, frames=50, repeat=False)
    plt.show()
    animator.save(
        "./GridAnalysisData/MultipleSpectrumAnimation-U_over_t{}-{}-{}".format(U_over_t01, U_over_t02, U_over_t03)
        + parameters + ".gif", writer=writer)
    plt.close(fig)


def multiple_current_and_spectrum_animation(U_over_t0s, parameters, params):
    """This is set up to work with any number of U/t0"""

    # necessary parameters
    lat = hhg(params.field, params.nup, params.ndown, params.nx, 0, 0, params.t)
    a_values = np.linspace(1, 10, 50)
    # axes to plot on for current expectation and power spectrum
    cx_ax = current_expectation((U_over_t0s[0] * params.t, 1), params, True)[0]
    sx_ax = current_expectation_power_spectrum((U_over_t0s[0] * params.t, 1), params, True)[0] / lat.freq

    # current data will be the length of the number of U/t0s passed in
    # each entry contains an array of current expectations for the given U/t0 over all a_values
    current_data = []
    for U_over_t0 in U_over_t0s:
        try:
            d = np.load("./GridAnalysisData/CurrentDensityExpectationAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy")
        except:
            d = np.array([current_expectation((U_over_t0 * params.t, a), params) for a in a_values])
            np.save("./GridAnalysisData/CurrentDensityExpectationAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy", d)
        current_data.append(d)

    # spectrum data will be the length of the number of U/t0s passed in
    # each entry contains an array of spectra for the given U/t0 over all a_values
    spectrum_data = []
    for U_over_t0 in U_over_t0s:
        try:
            d = np.load("./GridAnalysisData/CurrentSpectrumAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy")
        except:
            d = np.array([current_expectation_power_spectrum((U_over_t0 * params.t, a), params) for a in a_values])
            np.save("./GridAnalysisData/CurrentSpectrumAnimationData-U_over_t{}".format(
                U_over_t0) + parameters + ".npy", d)
        spectrum_data.append(d)

    # the number of unique comparisons we can make is 1 + 2 + ... + |U_over_t0s| - 1
    # below we have lists for the time domain cost and spectra cost
    # each list is the length of the number of a values and each element
    # is the length of the number of unique comparisons
    scatterx = []
    scattery = []
    labels = []  # labels for plotting points, length is number of unique comparisons
    for i in range(len(a_values)):
        temp_time = []
        temp_freq = []
        for j in range(len(U_over_t0s)):
            for k in range(len(U_over_t0s)):
                # avoid repeats and comparisons with the same system
                if j < k:
                    temp_time.append(time_domain_objective((current_data[j][i], current_data[k][i])))
                    temp_freq.append(objective_w_spectrum((spectrum_data[j][i], spectrum_data[k][i])))
                    # we only need to get the labels once
                    if i == 0:
                        labels.append("$({} \\cdot t_0, {} \\cdot t_0)$".format(U_over_t0s[j], U_over_t0s[k]))
        temp_time = np.array(temp_time)
        temp_freq = np.array(temp_freq)
        scatterx.append(temp_time)
        scattery.append(temp_freq)
    scatterx = np.array(scatterx)
    scatterx /= scatterx.max()
    scattery = np.array(scattery)
    scattery /= scattery.max()

    # now for plotting, axs[0] is for current expectations,
    # axs[1] is for scatter plot cost comparison, and axs[2]
    # is for current power spectrum
    writer = ani.PillowWriter()
    fig, axs = plt.subplots(3)
    fig.set_figheight(13.5)
    fig.set_figwidth(12)
    title = plt.suptitle("$a = 1.000$")

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Current Density")
    axs[1].set_xlabel("Time Domain Cost")
    axs[1].set_ylabel("Frequency Domain Cost")
    axs[0].set_title("Current Density Expectation")
    axs[2].set_xlabel("Harmonic Order")
    axs[2].set_ylabel("Power")
    axs[2].set_title("Current Power Spectrum")
    current_plots = []
    spectrum_plots = []
    for i in range(len(U_over_t0s)):
        cplot, = axs[0].plot(cx_ax, current_data[i][0], label="$U = {:.2f} \\cdot t_0$".format(U_over_t0s[i]))
        current_plots.append(cplot)
        splot, = axs[2].semilogy(sx_ax, spectrum_data[i][0], label="$U = {:.2f} \\cdot t_0$".format(U_over_t0s[i]))
        spectrum_plots.append(splot)

    scatter_plots = []
    # loop over number of unique comparisons
    for i in range(sum(range(len(U_over_t0s)))):
        sctr, = axs[1].plot(scatterx[0][i], scattery[0][i], ".", label=labels[i])
        scatter_plots.append(sctr)

    axs[0].legend(loc="upper right")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend(loc="upper left")
    axs[2].set_xlim(0, 55)
    axs[2].legend(loc="upper right")

    def chart(num):
        title.set_text("$a = {:.3f}$".format(1 + (9 / 49) * num))
        for i in range(len(U_over_t0s)):
            current_plots[i].set_ydata(current_data[i][num])
            spectrum_plots[i].set_ydata(spectrum_data[i][num])

        for i in range(sum(range(len(U_over_t0s)))):
            scatter_plots[i].set_xdata(scatterx[num][i])
            scatter_plots[i].set_ydata(scattery[num][i])

    animator = ani.FuncAnimation(fig, chart, frames=50, repeat=False)
    plt.show()
    filename = "./GridAnalysisData/MultipleCurrentandSpectraAnimation-U_over_t"
    for U_over_t0 in U_over_t0s[:-1]:
        filename += str(U_over_t0) + "-"
    filename += str(U_over_t0s[-1]) + parameters + ".gif"
    animator.save(filename, writer=writer)
    plt.close(fig)


sites = 10
sym = True

params = set_up(sym, sites)

U_over_t0_min = 0
U_over_t0_max = 10
a_min = 1
a_max = 10

# by default, we should be looking over a regular grid with cut spectra
random_points = False
cut_spectra = False

parameters = '-sites{}-U_min{}t0-U_max{}t0-{}a_min-{}a_max'.format(params.nx, U_over_t0_min, U_over_t0_max, a_min,
                                                                   a_max)
# these parameters should only be used when a calculation involves spectra
spectrum_parameters = ""
if sym:
    parameters += '-withsymmetry'
else:
    parameters += '-withoutsymmetry'

if random_points:
    parameters += '-randompoints'

if not cut_spectra:
    spectrum_parameters += '-without_cuts'

x_vals, bounds, U_vals, a_vals = get_data_points(params, U_over_t0_min, U_over_t0_max, a_min, a_max, random_points)

"""Spectral Analysis"""
if cut_spectra:
    spectra = get_spectra(params, x_vals, parameters + spectrum_parameters)
else:
    spectra = get_spectra_wo_cuts(params, x_vals, parameters + spectrum_parameters)
spectra = np.load('./GridAnalysisData/Spectra' + parameters + spectrum_parameters + '.npy')
print(spectra.shape)
# cost_comparison(spectra, parameters + spectrum_parameters, "frequency")
# frequency_cost_comparison = np.load('./GridAnalysisData/SpectralCostComparisonData' + parameters
#                                     + spectrum_parameters + '.npy')
# graph_cost_comparison(cost_comparison, parameters + spectrum_parameters, "frequency")
# ani_data_freq = animation_data(spectra, parameters + spectrum_parameters, "frequency")
# ani_data_freq = np.load("./GridAnalysisData/SpectralAnimationData" + parameters + spectrum_parameters + ".npy")
# new_ani_data_freq = animation_data(spectra, parameters + spectrum_parameters, "new frequency")
# new_ani_data_freq = np.load("./GridAnalysisData/NewSpectralAnimationData" + parameters + spectrum_parameters + ".npy")
# animation(ani_data, parameters + spectrum_parameters, "frequency")
# spectrum_animation(5, parameters + spectrum_parameters, params)
# multiple_spectrum_animation([.5,1,5], parameters + spectrum_parameters, params)

"""Time Domain Analysis"""
# expecs = get_current_expecs(params, x_vals, parameters)
# expecs = np.load('./GridAnalysisData/CurrentExpectations' + parameters + '.npy')
# cost_comparison(expecs, parameters, "time")
# time_cost_comparison = np.load('./GridAnalysisData/TimeDomainCostComparisonData' + parameters + '.npy')
# cost_comparison(expecs, parameters, "new time")
# new_time_cost_comparison = np.load('./GridAnalysisData/TimeDomainNewCostComparisonData' + parameters + '.npy')
# graph_cost_comparison(cost_comparison, parameters, "time")
# ani_data_time = animation_data(expecs, parameters, "time")
ani_data_time = np.load("./GridAnalysisData/TimeDomainAnimationData" + parameters + ".npy")
# new_ani_data_time = animation_data(expecs, parameters, "new time")
# new_ani_data_time = np.load("./GridAnalysisData/NewTimeDomainAnimationData" + parameters + ".npy")
# animation(ani_data_time, parameters, "time")
# current_animation(7, parameters, params)
# multiple_current_animation([.5, 1, 5], parameters + spectrum_parameters, params)

"""Scatter plot"""
# scatter_plot_animation(ani_data_time, ani_data_freq, parameters + spectrum_parameters)
# scatter_plot_animation_new(ani_data_time, ani_data_freq, parameters)
# scatter_plot_animation(new_ani_data_time, new_ani_data_freq, "NewCost" + parameters)

# multiple_current_and_spectrum_animation([0, .5, 5], parameters + spectrum_parameters, params)

# scatter_plot_field(params, parameters)
# scatter_plot_cycles(params, parameters)

# paper_graph(ani_data_time, ani_data_freq)
# paper_graph_F0(ani_data_time, ani_data_freq)


def other_paper_graph():
    nsites = 10
    symmetry = True

    params2 = set_up(symmetry, nsites)

    a1 = 5
    a2 = 10

    U1 = 0
    U2 = 2
    U3 = 10

    f, p1 = current_expectation_power_spectrum([U1 * params2.t, a1], params2, True)
    f2, p2 = current_expectation_power_spectrum([U2 * params2.t, a1], params2, True)
    f5, p5 = current_expectation_power_spectrum([U3 * params2.t, a1], params2, True)
    f3, p3 = current_expectation_power_spectrum([U1 * params2.t, a2], params2, True)
    f4, p4 = current_expectation_power_spectrum([U2 * params2.t, a2], params2, True)
    f6, p6 = current_expectation_power_spectrum([U3 * params2.t, a2], params2, True)

    fig = plt.figure()
    fig.clear()
    fig.set_figwidth(4)
    fig.set_figheight(5)
    ax1, ax2 = fig.subplots(2, sharex=True)
    plt.xlabel("Harmonic Order")
    ax1.text(.9, .98, '(a)', fontsize='x-large', verticalalignment='top', transform=ax1.transAxes)
    ax2.text(.9, .98, '(b)', fontsize='x-large', verticalalignment='top', transform=ax2.transAxes)

    lat = hhg(field=params2.field, nup=params2.nup, ndown=params2.ndown, nx=params2.nx, ny=0, U=0, t=params2.t,
                  F0=params2.F0, a=1, pbc=params2.pbc)

    f /= lat.freq
    ax1.semilogy(f, p1, label="$U = {} \\cdot t_0$".format(U1), color="blue")
    ax1.semilogy(f, p2, label="$U = {} \\cdot t_0$".format(U2), color="green")
    ax1.semilogy(f, p5, label="$U = {} \\cdot t_0$".format(U3), color="orange")
    ax2.semilogy(f, p3, label="$U = {} \\cdot t_0$".format(U1), color="blue")
    ax2.semilogy(f, p4, label="$U = {} \\cdot t_0$".format(U2), color="green")
    ax2.semilogy(f, p6, label="$U = {} \\cdot t_0$".format(U3), color="orange")
    ax1.set_ylabel("Power")
    ax2.set_ylabel("Power")

    # ax1.set_yticks([])
    # ax2.set_yticks([])
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower left")

    ax1.set_xlim(0, 70)
    # ax2.set_xlim(0, 30)

    # plt.text(0, 2.1, '(a)', fontsize='x-large')

    plt.show()


# other_paper_graph()


def spectra_single_plot():
    U1 = 0 * params.t
    U2 = 0.05 * params.t
    U3 = 10 * params.t
    a = 10

    f, p1 = current_expectation_power_spectrum([U1, a], params, True)
    # p2 = current_expectation_power_spectrum([U2, a], params, False)
    # p3 = current_expectation_power_spectrum([U3, a], params, False)

    lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=0, t=params.t,
                  F0=params.F0, a=1, pbc=params.pbc)

    f /= lat.freq

    plt.semilogy(f, p1, label="$U = {} \\cdot t_0$".format(U1 / params.t))
    # plt.text(1, .1, "$\\omega_0$", color="black", fontsize="x-large")
    # plt.text(3, .1, "$3\\omega_0$", color="black", fontsize="x-large")
    # plt.text(5, .003, "$5\\omega_0$", color="black", fontsize="x-large")
    # plt.text(7, 2e-5, "$7\\omega_0$", color="black", fontsize="x-large")
    # plt.semilogy(f, p2, label="$U = {} \\cdot t_0$".format(U2 / params.t))
    # plt.semilogy(f, p3, label="$U = {} \\cdot t_0$".format(U3 / params.t))
    plt.xlim(0, 30)
    plt.ylim(1e-20, None)
    # plt.axis('off')
    plt.legend()
    plt.xlabel("Harmonic Order")
    plt.ylabel("Power")
    plt.show()


# spectra_single_plot()


def current_single_plot():
    U1 = 0 * params.t
    U2 = 0.1 * params.t
    U3 = 5 * params.t
    a = 2

    times, c1 = current_expectation([U1, a], params, True)
    c2 = current_expectation([U2, a], params, False)
    c3 = current_expectation([U3, a], params, False)

    plt.plot(times, c1, label="$U = {} \\cdot t_0$".format(U1 / params.t))
    # plt.text(1, .1, "$\\omega_0$", color="black", fontsize="x-large")
    # plt.text(3, .1, "$3\\omega_0$", color="black", fontsize="x-large")
    # plt.text(5, .003, "$5\\omega_0$", color="black", fontsize="x-large")
    # plt.text(7, 2e-5, "$7\\omega_0$", color="black", fontsize="x-large")
    plt.plot(times, c2, label="$U = {} \\cdot t_0$".format(U2 / params.t))
    plt.plot(times, c3, label="$U = {} \\cdot t_0$".format(U3 / params.t))
    # plt.axis('off')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Current Density Expectation")
    plt.show()


# current_single_plot()


def cost_function_animation_time():
    U1 = 4 * params.t
    U2 = 5 * params.t
    a = 4

    times, c1 = current_expectation([U1, a], params, True)
    c2 = current_expectation([U2, a], params, False)

    writer = ani.PillowWriter(fps=400)
    fig, ax = plt.subplots()

    def chart(i):
        ax.clear()
        ax.plot(times[i:], c1[i:], color="blue", label="$\\langle \\hat{J}_i(t) \\rangle$")
        ax.plot(times[i:], c2[i:], color="red", label="$\\langle \\hat{J}_j(t) \\rangle$")
        ax.plot(times[:i], abs(c1[:i] - c2[:i]), color="purple",
                label="$|\\langle \\hat{J}_i(t) \\rangle - \\langle \\hat{J}_j(t) \\rangle|$")
        ax.legend()

    animator = ani.FuncAnimation(fig, chart, frames=len(times), interval=0.5, repeat=False)
    # plt.show()
    animator.save("./PaperImages/TimeCostExperiment.gif", writer=writer)
    plt.close(fig)


# cost_function_animation_time()


def cost_function_animation_spectra():
    U1 = 4 * params.t
    U2 = 5 * params.t
    a = 4

    f, p1 = current_expectation_power_spectrum([U1, a], params, True)
    p2 = current_expectation_power_spectrum([U2, a], params, False)

    writer = ani.PillowWriter()
    fig, ax = plt.subplots()

    lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=0, t=params.t,
                  F0=params.F0, a=1, pbc=params.pbc)

    f /= lat.freq

    inds = np.where(f <= 30)[0]
    f = f[inds]
    p1 = p1[inds]
    p2 = p2[inds]

    ax.set_xlabel("Harmonic Order")
    def chart(i):
        ax.clear()
        ax.semilogy(f[i:], p1[i:], color="blue", label="$S_i(\\omega)$")
        ax.semilogy(f[i:], p2[i:], color="red", label="$S_i(\\omega)$")
        ax.semilogy(f[:i], abs(p1[:i] - p2[:i]), color="purple",
                label="$|S_i(\\omega) - S_i(\\omega)$")
        ax.legend()

    animator = ani.FuncAnimation(fig, chart, frames=len(f), interval=0.5, repeat=False)
    # plt.show()
    animator.save("./PaperImages/SpectraCostExperiment.gif", writer=writer)
    plt.close(fig)


# cost_function_animation_spectra()
