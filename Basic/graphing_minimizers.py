import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""Hubbard model Parameters"""
L = 6 # system size
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

parameters = '-{}sites-{}t0-{}U-{}a-{}field-{}amplitude-{}cycles-{}steps-{}pbc'.format(L,t0,target_U,target_a,field,F0,10,2000,pbc)

"""Graphing the custom global minimizer"""
initial_Us = np.load('./GlobalMinimizeData/initial_Us'+parameters+'.npy')
initial_Us = np.reshape(initial_Us, (5,5)) / target_U
initial_as = np.load('./GlobalMinimizeData/initial_as'+parameters+'.npy')
initial_as = np.reshape(initial_as, (5,5)) / target_a
cost = np.load('./GlobalMinimizeData/cost'+parameters+'.npy')
cost = np.reshape(cost, (5,5))
runtimes = np.load('./GlobalMinimizeData/runtimes'+parameters+'.npy')
runtimes = np.reshape(runtimes, (5,5)) / 3600
final_Us = np.load('./GlobalMinimizeData/final_Us'+parameters+'.npy')
final_Us = np.reshape(final_Us, (5,5)) / t0
final_as = np.load('./GlobalMinimizeData/final_as'+parameters+'.npy')
final_as = np.reshape(final_as, (5,5))

print(final_Us)
print(final_as)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_wireframe(initial_Us, initial_as, runtimes)
#
# plt.show()

