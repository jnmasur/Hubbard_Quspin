import numpy as np
from evolve import phi_tl


def H_expec(psis, times, onsite, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    """
    Calculates expectation of the hamiltonian
    :param psis: list of states at every point in the time evolution
    :param times: the times at which psi was calculated
    :param phi_func: the function used to calculate phi
    :return: an array of the expectation values of a Hamiltonian
    """
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        phi = phi_func(current_time, lat, cycles)
        # H|psi>
        Hpsi = -lat.t * (np.exp(-1j*phi) * hop_left.dot(psi) + np.exp(1j*phi) * hop_right.dot(psi)) + \
            lat.U * onsite.dot(psi)
        # <psi|H|psi>
        expec.append((np.vdot(psi, Hpsi)).real)
    return np.array(expec)


def J_expec(psis, times, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    """
    Calculates expectation of the current density
    :param psis: list of states at every point in the time evolution
    :param times: the times at which psi was calculated
    :param phi_func: the function used to calculate phi
    :return: an array of the expectation values of a density
    """
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        phi = phi_func(current_time, lat, cycles)
        # J|psi>
        Jpsi = -1j*lat.a*lat.t* (np.exp(-1j*phi) * hop_left.dot(psi) - np.exp(1j*phi) * hop_right.dot(psi))
        # <psi|J|psi>
        expec.append((np.vdot(psi, Jpsi)).real)
    return np.array(expec)

def dH_dU_expec(psis, dpsis_dU, times, onsite, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        dpsi_dU = dpsis_dU[:,i]
        phi = phi_func(current_time, lat, cycles)
        # H|psi>
        Hpsi = -lat.t * (np.exp(-1j * phi) * hop_left.dot(psi) + np.exp(1j * phi) * hop_right.dot(psi)) + \
               lat.U * onsite.dot(psi)
        # <dpsi/du|H|psi>
        a = np.vdot(dpsi_dU, Hpsi)
        # (dH/dU)|psi>
        dH_dU_psi = onsite.dot(psi)
        # <psi|(dH/dU)|psi>
        b = np.vdot(psi, dH_dU_psi)
        # <dpsi/du|H|psi> + <psi|(dH/dU)|psi> + <psi|H|dpsi/dU>
        expec.append((a + b + a.conj()).real)
    return np.array(expec)

def dH_da_expec(psis, dpsis_da, times, onsite, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        dpsi_da = dpsis_da[:,i]
        phi = phi_func(current_time, lat, cycles)
        # H|psi>
        Hpsi = -lat.t * (np.exp(-1j * phi) * hop_left.dot(psi) + np.exp(1j * phi) * hop_right.dot(psi)) + \
               lat.U * onsite.dot(psi)
        # <dpsi/da|H|psi>
        a = np.vdot(dpsi_da, Hpsi)
        # (dH/da)|psi>
        dH_da_psi = 1j*lat.t*(phi/lat.a) * (np.exp(-1j*phi) * hop_left.dot(psi) - np.exp(1j*phi) * hop_right.dot(psi))
        # <psi|(dH/da)|psi>
        b = np.vdot(psi, dH_da_psi)
        # <dpsi/da|H|psi> + <psi|(dH/da)|psi> + <psi|H|dpsi/da>
        expec.append((a + b + a.conj()).real)
    return np.array(expec)

def dJ_dU_expec(psis, dpsis_dU, times, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        dpsi_dU = dpsis_dU[:,i]
        phi = phi_func(current_time, lat, cycles)
        # J|psi>
        Jpsi = -1j*lat.a*lat.t* (np.exp(-1j*phi) * hop_left.dot(psi) - np.exp(1j*phi) * hop_right.dot(psi))
        # <dpsi/dU|J|psi>
        a = np.vdot(dpsi_dU, Jpsi)
        # <dpsi/dU|J|psi> + <psi|J|dpsi/dU>
        expec.append((a + a.conj()).real)
    return np.array(expec)

def dJ_da_expec(psis, dpsis_da, times, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    expec = []
    for i in range(len(times)):
        current_time = times[i]
        psi = psis[:,i]
        dpsi_da = dpsis_da[:,i]
        phi = phi_func(current_time, lat, cycles)
        # J|psi>
        Jpsi = -1j * lat.a * lat.t * (np.exp(-1j * phi) * hop_left.dot(psi) - np.exp(1j * phi) * hop_right.dot(psi))
        # <dpsi/da|J|psi>
        a = np.vdot(dpsi_da, Jpsi)
        # (dJ/da)|psi>
        dJ_da_psi = lat.t * ((-1j - phi)*np.exp(-1j*phi)*hop_left.dot(psi) + (1j - phi)*np.exp(1j*phi)*hop_right.dot(psi))
        # <psi|(dJ/da)|psi>
        b = np.vdot(psi, dJ_da_psi)
        # <dpsi/da|J|psi> + <psi|(dJ/da)|psi> + <psi|J|dpsi/da>
        expec.append((a + b + a.conj()).real)
    return np.array(expec)
