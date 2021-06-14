from quspin.operators import hamiltonian  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
import quspin.tools.evolution as evolution
import numpy as np  # general math functions
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeResult, minimize
from scipy import signal
import expectations as expec
from tools import parameter_instantiate as hhg  # Used for scaling units.
from evolve import evolve_psi


def spectrum(current, delta):
    """
    Gets power spectrum of the current
    :param current: the induced current in the lattice
    :param delta: timestep between current points
    :return: the power spectrum of the current
    """
    at = np.gradient(current, delta)
    return spectrum_welch(at, delta)


def spectrum_welch(at, delta):
    return signal.welch(at, 1. / delta, nperseg=len(at), scaling='spectrum')


"""This class contains all unscaled parameters for passing into the objective function"""


# note U and a are the optimization parameters, so they are not contained in this class
class Parameters:
    def __init__(self, nx, nup, ndown, t0, field, F0, pbc):
        self.nx = nx
        self.nup = nup
        self.ndown = ndown
        self.t = t0
        self.pbc = pbc
        self.field = field
        self.F0 = F0
        self.basis = None

    def set_basis(self, symmetry=True):
        """
        This method sets the basis for our system, we can use some symmetries of the system
        to reduce the size of the Hilbert Space
        """
        if symmetry:
            self.basis = spinful_fermion_basis_1d(self.nx, Nf=(self.nup, self.ndown), sblock=1, kblock=1)
        else:
            self.basis = spinful_fermion_basis_1d(self.nx, Nf=(self.nup, self.ndown))


def objective(x, J_target, params, graph=False, fname=None):
    """
    Our objective function that we want to minimize:
    E(U,a) = int(JT - <J>)^2
    :param x: input array [U,a]
    :param J_target: tuple of (frequencies, spectrum)
    :param params: an instance of Parameters class
    :param graph: if True, the target and calculated current will be graphed together
    :param fname: name of the file to save the graph to
    :return: The cost of the function
    """

    # optimization parameters
    oU = x[0]
    oa = x[1]

    # unpack J_target
    target_freqs, target_spectrum = J_target

    # contains all important variables
    lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=oU, t=params.t,
              F0=params.F0, a=oa, pbc=params.pbc)

    # gets times to evaluate at
    cycles = 10
    n_steps = 2000
    start = 0
    stop = cycles / lat.freq
    times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

    int_list = [[1.0, i, i] for i in range(params.nx)]
    static_Hamiltonian_list = [
        ["n|n", int_list]  # onsite interaction
    ]
    # n_j,up n_j,down
    onsite = hamiltonian(static_Hamiltonian_list, [], basis=params.basis, **no_checks)

    hop = [[1.0, i, i + 1] for i in range(params.nx - 1)]
    if lat.pbc:
        hop.append([1.0, params.nx - 1, 0])

    # c^dag_j,sigma c_j+1,sigma
    hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=params.basis, **no_checks)
    # c^dag_j+1,sigma c_j,sigma
    hop_right = hop_left.getH()

    H = -lat.t * (hop_left + hop_right) + lat.U * onsite

    """build ground state"""
    E, psi_0 = H.eigsh(k=1, which='SA')
    psi_0 = np.squeeze(psi_0)

    """evolve the system"""
    psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=(onsite, hop_left, hop_right, lat, cycles))
    psi_t = np.squeeze(psi_t)
    # get the expectation value of J
    J_expec = expec.J_expec(psi_t, times, hop_left, hop_right, lat, cycles)

    expec_freqs, J_expec_spectrum = spectrum(J_expec, delta)

    fval = np.linalg.norm(np.log10(target_spectrum) - np.log10(J_expec_spectrum))

    # just some graphing stuff
    if graph:
        plt.plot(J_expec_spectrum, color='red', label='$\\langle\\hat{J}(t)\\rangle$')
        plt.plot(target_spectrum, color='blue', linestyle='dashed', label='$J^{T}(t)$')
        plt.legend(loc='upper left')
        plt.xlabel("Time")
        plt.ylabel("Current")
        plt.title("Target Current vs Best Fit Current")
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    # print("Fval =", fval, "for x =", x)

    return fval


def objective_w_spectrum(target_and_res):
    """
    Calculates cost given both spectra
    """
    target, res = target_and_res
    min_val = 1e-20
    indx = np.where((target > min_val) & (res > min_val))[0]
    if len(indx) > 10:
        target /= target.max()
        res /= res.max()
        return np.linalg.norm(np.log10(target[indx]) - np.log10(res[indx]))
    else:
        return np.nan


def new_objective(both):

    target, res = both
    target /= target.max()
    res /= res.max()
    return np.linalg.norm(target - res)


def time_domain_objective(both):
    current1, current2 = both
    current1 /= current1.max()
    current2 /= current2.max()
    return np.linalg.norm(current1 - current2)
    # return np.trapz((current1 - current2)**2)


def new_time_domain_objective(both):
    current1, current2 = both
    current1 /= current1.max()
    current2 /= current2.max()
    return np.linalg.norm(np.log10(abs(current1 - current2)))


def get_seeds(size):
    """
    Generate unique random seeds for seeding them into random number generators in multiprocessing simulations
    This utility is to avoid the following artifact:
        https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
    :param size: number of samples to generate
    :return: numpy.array of np.uint32
    """
    # Note that np.random.seed accepts 32 bit unsigned integers

    # get the maximum value of np.uint32 can take
    max_val = np.iinfo(np.uint32).max

    # A set of unique and random np.uint32
    seeds = set()

    # generate random numbers until we have sufficiently many nonrepeating numbers
    while len(seeds) < size:
        seeds.update(
            np.random.randint(max_val, size=size, dtype=np.uint32)
        )

    # make sure we do not return more numbers that we are asked for
    return np.fromiter(seeds, np.uint32, size)


def randomize_parameters(best_x, rU, ra, bounds, seed=None):
    """
    Used to randomize parameters to be passed into minimize for our global minimizer.
    :param best_x: the current best x
    :param rU: determines the range of numbers that can be randomly generated for U
    :param ra: determines the range of numbers that can be randomly generated for a
    :param bounds: a tuple of tuples with bounds for U and a
    :param seed: the seed for numpy random
    :return: a numpy array containing the initial x to be minimized
    """

    # unpack x variable
    best_U, best_a = best_x

    # unpack bounds
    U_bounds, a_bounds = bounds
    U_lower, U_upper = U_bounds
    a_lower, a_upper = a_bounds

    # seed the generator if it is not None
    if seed is not None:
        np.random.seed(seed)

    # [U_lower, U_lower + rU)
    if best_U - U_lower < rU:
        U = rU * np.random.rand() + U_lower

    # [U_upper - rU, U_upper)
    elif U_upper - best_U < rU:
        U = rU * np.random.rand() + (U_upper - rU)

    # [U-rU/2, U+rU/2)
    else:
        U = (rU * (np.random.rand() - .5)) + best_U

    # [a_lower , a_lower + ra)
    if best_a - a_lower < ra:
        a = ra * np.random.rand() + a_lower

    # [a_upper - ra, a_upper)
    elif a_upper - best_a < ra:
        a = ra * np.random.rand() + (a_upper - ra)

    # [a-ra/2,a+ra/2)
    else:
        a = (ra * (np.random.rand() - .5)) + best_a

    return np.array([U, a])


def global_minimize_single_thread(x0, fun, J_target, params, minimizer=minimize):
    """
    Essentially BasinHopping, but written by yours truly
    Takes random steps from the x value and does local minimizations at that parameter
    :param x0: initial optimization parameters
    :param fun: function to optimize
    :param J_target: the target current
    :param params: instance of Parameters class
    :param minimizer: the local minimizer function
    :return: an instance of the OptimizeResult class with the best x value and function value
    """

    # radius of range for randomized parameters
    rU = .4 * params.t
    ra = .4

    # bounds for variables
    U_upper = 10 * params.t
    U_lower = 0
    a_upper = 10
    a_lower = 0

    # first_x0 = x0

    # get the first value of the function
    fval = fun(x0, J_target, params)
    best_fval = fval
    best_x = x0

    niter = 0
    # loop has 100 max iterations and terminates at fval < 1e-6
    while best_fval > 1e-6 and niter < 100:
        niter += 1
        res = minimizer(fun, x0, (J_target, params), bounds=((U_lower, U_upper), (a_lower, a_upper)))

        if res.fun < best_fval:
            best_fval = res.fun
            best_x = res.x

        best_U = best_x[0]
        best_a = best_x[1]
        # [U_lower, U_lower + rU)
        if best_U - U_lower < rU:
            x0[0] = rU * np.random.rand() + U_lower

        # [U_upper - rU, U_upper)
        elif U_upper - best_U < rU:
            x0[0] = rU * np.random.rand() + (U_upper - rU)

        # [U-rU/2, U+rU/2)
        else:
            x0[0] = (rU * (np.random.rand() - .5)) + best_U

        # [a_lower , a_lower + ra)
        if best_a - a_lower < ra:
            x0[1] = ra * np.random.rand() + a_lower

        # [a_upper - ra, a_upper)
        elif a_upper - best_a > ra:
            x0[1] = ra * np.random.rand() + (a_upper - ra)

        # [a-ra/2,a+ra/2)
        else:
            x0[1] = (ra * (np.random.rand() - .5)) + best_a

    return OptimizeResult(x=best_x, fun=best_fval)


# simply used to call minimize
def minimize_wrapper(args):
    fun, x0, J_target, params, bounds = args
    return minimize(fun, x0, args=(J_target, params), bounds=bounds, options={'ftol': 1e-10})


def current_expectation(x, params, return_time=False):
    U, a = x

    # contains all important variables
    lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=U, t=params.t,
              F0=params.F0, a=a, pbc=params.pbc)

    # gets times to evaluate at
    cycles = 10
    n_steps = 2000
    start = 0
    stop = cycles / lat.freq
    times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

    int_list = [[1.0, i, i] for i in range(params.nx)]
    static_Hamiltonian_list = [
        ["n|n", int_list]  # onsite interaction
    ]
    # n_j,up n_j,down
    onsite = hamiltonian(static_Hamiltonian_list, [], basis=params.basis, **no_checks)

    hop = [[1.0, i, i + 1] for i in range(params.nx - 1)]
    if lat.pbc:
        hop.append([1.0, params.nx - 1, 0])

    # c^dag_j,sigma c_j+1,sigma
    hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=params.basis, **no_checks)
    # c^dag_j+1,sigma c_j,sigma
    hop_right = hop_left.getH()

    H = -lat.t * (hop_left + hop_right) + lat.U * onsite

    """build ground state"""
    # print("calculating ground state")
    E, psi_0 = H.eigsh(k=1, which='SA')
    psi_0 = np.squeeze(psi_0)
    # print("ground state calculated, energy is {:.2f}".format(E[0]))

    # evolve the system
    psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=(onsite, hop_left, hop_right, lat, cycles))
    psi_t = np.squeeze(psi_t)
    # get the expectation value of J
    J_expec = expec.J_expec(psi_t, times, hop_left, hop_right, lat, cycles)

    if return_time:
        return times, J_expec
    else:
        return J_expec


def current_expectation_power_spectrum(x, params, return_freqs=False):
    """
    Returns the power spectrum of a current based on the parameters and x
    :param x: U, a
    :param params: all parameters
    :return: power spectrum of J
    """
    U, a = x

    lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=U, t=params.t,
              F0=params.F0, a=a, pbc=params.pbc)

    cycles = 10
    n_steps = 2000
    start = 0
    stop = cycles / lat.freq
    delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)[1]

    J = current_expectation(x, params)

    if return_freqs:
        return spectrum(J, delta)
    else:
        return spectrum(J, delta)[1]


class Result:
    """
    Simple class just to be used to return an optimized result
    """
    def __init__(self, x, fun, t):
        self.x = x
        self.fun = fun
        self.time = t

