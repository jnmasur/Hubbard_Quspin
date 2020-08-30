from quspin.operators import hamiltonian  # operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
import quspin.tools.evolution as evolution
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
from matplotlib import pyplot as plt
from scipy.integrate import trapz
import expectations as expec
from tools import parameter_instantiate as hhg  # Used for scaling units.
from evolve import evolve_psi

"""This class contains all unscaled parameters for passing into the objective and gradient functions"""
# note U and a are the optimization parameters, so they are not contained in this class
class Parameters:
    def __init__(self, nx, nup, ndown, t0, field, F0, pbc=True):
        self.nx = nx
        self.nup = nup
        self.ndown = ndown
        self.t = t0
        self.pbc = pbc
        self.field = field
        self.F0 = F0
        self.basis = None

    def set_basis(self):
        self.basis = spinful_fermion_basis_1d(self.nx, Nf=(self.nup, self.ndown))


"""The objective function that we want to minimize"""
"""E(U,a) = int(JT - <J>)^2"""
def objective(x, J_target, params, graph=False, x0=None):
    # optimization parameters
    oU = x[0]
    oa = x[1]

    # contains all important variables
    lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=oU, t=params.t, F0=params.F0, a=oa, pbc=params.pbc)

    # gets times to evaluate at
    cycles = 10
    n_steps = 2000
    start = 0
    stop = cycles / lat.freq
    times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

    if lat.U == 0:
        int_list = [[0.0, i, i] for i in range(params.nx)]
    else:
        int_list = [[1.0, i, i] for i in range(params.nx)]
    static_Hamiltonian_list = [
        ["n|n", int_list]  # onsite interaction
    ]
    # n_j,up n_j,down
    onsite = hamiltonian(static_Hamiltonian_list, [], basis=params.basis)

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    hop = [[1.0, i, i + 1] for i in range(params.nx - 1)]
    if lat.pbc:
        hop.append([1.0, params.nx - 1, 0])

    # c^dag_j,sigma c_j+1,sigma
    hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=params.basis, **no_checks)
    # c^dag_j+1,sigma c_j,sigma
    hop_right = hop_left.getH()

    H = -lat.t * (hop_left + hop_right) + lat.U * onsite

    """build ground state"""
    print("calculating ground state")
    E, psi_0 = H.eigsh(k=1, which='SA')
    psi_0 = np.squeeze(psi_0)
    print("ground state calculated, energy is {:.2f}".format(E[0]))

    psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=(onsite, hop_left, hop_right, lat, cycles))
    psi_t = np.squeeze(psi_t)

    J_expec = expec.J_expec(psi_t, times, hop_left, hop_right, lat, cycles)

    difference = J_target - J_expec

    fval = trapz(difference ** 2, dx=delta)

    # print("Function value:", fval)

    if graph:
        plt.plot(times, J_expec, color='red', label='$\\langle\\hat{J}(t)\\rangle$')
        plt.plot(times, J_target, color='blue', linestyle='dashed', label='$J^{T}(t)$')
        plt.legend(loc='upper left')
        plt.xlabel("Time")
        plt.ylabel("Current")
        plt.title("Target Current vs Best Fit Current")
        plt.show()
        parameters = ""
        if x0 is not None:
            parameters += "-{:.4f}U_initial-{:.4f}a_initial".format(x0[0],x0[1])
        parameters += "-{:.4f}U_final-{:.4f}a_final".format(x[0], x[1])
        parameters += '-{}sites-{}t0-{}field-{}amplitude-{}cycles-{}steps-{}pbc'.format(params.nx,params.t, params.field,params.F0,cycles,n_steps,params.pbc)
        plt.savefig("./MinimizedPlots/CurrentComparison"+parameters+'.pdf')

    return fval

"""returns gradient vector of function: <dE/dU, dE/da>"""
def J_gradient(x, J_target, params):

    # optimization parameters
    oU = x[0]
    oa = x[1]

    # contains all important variables
    lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=oU, t=params.t, F0=params.F0, a=oa, pbc=params.pbc)

    # +- dU/2 (we don't need the same for da, because it is only used for calculating the ground state)
    lat_plus_dU = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=oU+(params.dU / 2), t=params.t, F0=params.F0, a=oa, pbc=params.pbc)
    lat_minus_dU = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=oU-(params.dU / 2), t=params.t, F0=params.F0, a=oa, pbc=params.pbc)

    # these change after being scaled so we recalculate
    new_delta_U = lat_plus_dU.U - lat_minus_dU.U

    # gets times to evaluate at
    cycles = 10
    n_steps = 2000
    start = 0
    stop = cycles / lat.freq
    times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

    if lat.U == 0:
        int_list = [[0.0, i, i] for i in range(params.nx)]
    else:
        int_list = [[1.0, i, i] for i in range(params.nx)]
    static_Hamiltonian_list = [
        ["n|n", int_list]  # onsite interaction
    ]
    # n_j,up n_j,down
    onsite = hamiltonian(static_Hamiltonian_list, [], basis=params.basis)

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    hop = [[1.0, i, i + 1] for i in range(params.nx - 1)]
    if lat.pbc:
        hop.append([1.0, params.nx - 1, 0])

    # c^dag_j,sigma c_j+1,sigma
    hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=params.basis, **no_checks)
    # c^dag_j+1,sigma c_j,sigma
    hop_right = hop_left.getH()

    H = -lat.t * (hop_left + hop_right) + lat.U * onsite
    H_plus_dU = -lat_plus_dU.t * (hop_left + hop_right) + lat_plus_dU.U * onsite
    H_minus_dU = -lat_minus_dU.t * (hop_left + hop_right) + lat_minus_dU.U * onsite

    """build ground state"""
    print("calculating ground state")
    E, psi_0 = H.eigsh(k=1, which='SA')
    psi_0 = np.squeeze(psi_0)
    print("ground state calculated, energy is {:.2f}".format(E[0]))

    """build ground state for derivatives"""
    z, psi_plus_dU = H_plus_dU.eigsh(k=1, which='SA')
    z, psi_minus_dU = H_minus_dU.eigsh(k=1, which='SA')
    dpsi_dU_0 = (psi_plus_dU - psi_minus_dU)/new_delta_U
    dpsi_dU_0 = np.squeeze(dpsi_dU_0)
    dpsi_dU_0 = dpsi_dU_0 / np.linalg.norm(dpsi_dU_0)

    dpsi_da_0 = np.zeros((psi_0.shape)[0])

    big_psi = np.concatenate([psi_0, dpsi_dU_0, dpsi_da_0])
    print('evolving system')
    ti = time()
    # evolve the system
    psis_t = evolution.evolve(big_psi, 0.0, times, big_evolve, f_params=(onsite, hop_left, hop_right, lat, cycles))
    psis_t = np.squeeze(psis_t)
    print("Evolution done! This one took {:.2f} seconds".format(time() - ti))

    psi_t = psis_t[:int(len(psis_t) / 3)]
    dpsi_dU_t = psis_t[int(len(psis_t) / 3):int(2 * len(psis_t) / 3)]
    dpsi_da_t = psis_t[int(2 * len(psis_t) / 3):]

    J_expec = expec.J_expec(psi_t, times, hop_left, hop_right, lat, cycles)
    dJ_dU_expec = expec.dJ_dU_expec(psi_t, dpsi_dU_t, times, hop_left, hop_right, lat, cycles)
    dJ_da_expec = expec.dJ_da_expec(psi_t, dpsi_da_t, times, hop_left, hop_right, lat, cycles)

    difference = J_target - J_expec
    de_dU = difference * dJ_dU_expec
    de_da = difference * dJ_da_expec

    grad = np.array([-2 * trapz(de_dU, dx=delta), -2 * trapz(de_da, dx=delta)])
    print("CURRENT GRADIENT", grad)
    print("At U = {} and a = {}".format(oU, oa))

    return grad

def residuals(x, J_target, params):
    # optimization parameters
    oU = x[0]
    oa = x[1]

    # contains all important variables
    lat = hhg(field=params.field, nup=params.nup, ndown=params.ndown, nx=params.nx, ny=0, U=oU, t=params.t, F0=params.F0, a=oa, pbc=params.pbc)

    # gets times to evaluate at
    cycles = 10
    n_steps = 2000
    start = 0
    stop = cycles / lat.freq
    times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

    if lat.U == 0:
        int_list = [[0.0, i, i] for i in range(params.nx)]
    else:
        int_list = [[1.0, i, i] for i in range(params.nx)]
    static_Hamiltonian_list = [
        ["n|n", int_list]  # onsite interaction
    ]
    # n_j,up n_j,down
    onsite = hamiltonian(static_Hamiltonian_list, [], basis=params.basis)

    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    hop = [[1.0, i, i + 1] for i in range(params.nx - 1)]
    if lat.pbc:
        hop.append([1.0, params.nx - 1, 0])

    # c^dag_j,sigma c_j+1,sigma
    hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=params.basis, **no_checks)
    # c^dag_j+1,sigma c_j,sigma
    hop_right = hop_left.getH()

    H = -lat.t * (hop_left + hop_right) + lat.U * onsite

    """build ground state"""
    print("calculating ground state")
    E, psi_0 = H.eigsh(k=1, which='SA')
    psi_0 = np.squeeze(psi_0)
    print("ground state calculated, energy is {:.2f}".format(E[0]))

    psi_t = evolution.evolve(psi_0, 0.0, times, evolve_psi, f_params=(onsite, hop_left, hop_right, lat, cycles))
    psi_t = np.squeeze(psi_t)

    J_expec = expec.J_expec(psi_t, times, hop_left, hop_right, lat, cycles)

    difference = np.array(J_target - J_expec)

    return difference


def end_minimization(x,f,accept):
    if abs(f) < 10**-8:
        return True
    else:
        return None

def acceptance_test(f_new, x_new, f_old, x_old):
    if abs(f_new) < 10**-8:
        return "force accept"
    else:
        return True

