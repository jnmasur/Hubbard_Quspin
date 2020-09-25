import numpy as np

def phi_tl(current_time, lat, cycles):
    """
    Calculates phi
    :param current_time: time in the evolution
    :return: phi
    """
    phi = (lat.a * lat.F0 / lat.field) * (np.sin(lat.field * current_time / (2. * cycles)) ** 2.) * np.sin(
        lat.field * current_time)
    return phi

def evolve_psi(current_time, psi, onsite, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    """
    Evolves psi
    :param current_time: time in evolution
    :param psi: the current wavefunction
    :param phi_func: the function used to calculate phi
    :return: -i * H|psi>
    """

    # print("Simulation Progress: |" + "#" * int(current_time * lat.freq) + " " * (10 - int(current_time * lat.freq))
    #       + "|" + "{:.2f}".format(current_time * lat.freq * 10) + "%", end="\r")

    phi = phi_func(current_time, lat, cycles)

    a = -1j * (-lat.t * (np.exp(-1j*phi)*hop_left.static.dot(psi) + np.exp(1j*phi)*hop_right.static.dot(psi))
               + lat.U * onsite.static.dot(psi))

    return a

"""This function evolves psi, dpsi/dU dpsi/da"""
def big_evolve(current_time, big_psi, onsite, hop_left, hop_right, lat, cycles, phi_func=phi_tl):
    psi = big_psi[:int(len(big_psi)/3)]
    dpsi_dU = big_psi[int(len(big_psi)/3):int(2*len(big_psi)/3)]
    dpsi_da = big_psi[int(2*len(big_psi)/3):]

    print("Simulation Progress: |" + "#"*int(current_time*lat.freq) + " "*(10-int(current_time*lat.freq)) + "|"
          + "{:.2f}".format(current_time*lat.freq*10)+"%", end="\r")

    phi = phi_func(current_time, lat, cycles)
    expiphi = np.exp(1j*phi)
    expiphiconj = np.exp(-1j*phi)

    on_psi = onsite.static.dot(psi)
    left_psi = hop_left.static.dot(psi)
    right_psi = hop_right.static.dot(psi)

    left = expiphiconj * left_psi
    right = expiphi * right_psi

    # H|psi>
    a = -1j * (-lat.t * (left + right) + lat.U * on_psi)

    # H|dpsi/dU> + (dH/dU)|psi>
    b = -1j * (-lat.t * (expiphiconj * hop_left.static.dot(dpsi_dU) + expiphi * hop_right.static.dot(dpsi_dU)) + lat.U * onsite.static.dot(dpsi_dU))
    b += -1j * on_psi

    # H|dpsi/da> + (dH_da)|psi>
    c = -1j * (-lat.t * (expiphiconj * hop_left.static.dot(dpsi_da) + expiphi * hop_right.static.dot(dpsi_da)) + lat.U * onsite.static.dot(dpsi_da))
    c += -1j * (1j * lat.t * (phi/lat.a) * (left - right))

    return np.concatenate([a,b,c])

"""This function evolves 2 wavefunctions simultaneously for calculating H(x+delta x) - H(x-delta x)"""
def evolve_cfd(current_time, psi_cfd, onsite, hop_left, hop_right, lat_plus, lat_minus, cycles, phi_func=phi_tl):
    psi_plus = psi_cfd[:int(len(psi_cfd)/2)]
    psi_minus = psi_cfd[int(len(psi_cfd)/2):]

    print("Simulation Progress: |" + "#"*int(current_time*lat_plus.freq) + " "*(10-int(current_time*lat_plus.freq)) + "|"
          + "{:.2f}".format(current_time*lat_plus.freq*10)+"%", end="\r")

    phi_plus = phi_func(current_time, lat_plus, cycles)
    phi_minus = phi_func(current_time, lat_minus, cycles)

    # H+|psi+>
    a = -1j * (-lat_plus.t * (np.exp(-1j*phi_plus) * hop_left.static.dot(psi_plus) + np.exp(1j*phi_plus)
                              * hop_right.static.dot(psi_plus)) + lat_plus.U * onsite.static.dot(psi_plus))

    # H-|psi->
    b = -1j * (-lat_minus.t * (np.exp(-1j*phi_minus) * hop_left.static.dot(psi_minus) + np.exp(1j*phi_minus)
                              * hop_right.static.dot(psi_minus)) + lat_minus.U * onsite.static.dot(psi_minus))

    return np.concatenate([a,b])
