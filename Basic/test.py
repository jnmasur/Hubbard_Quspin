import numpy as np

threads_per_x0 = 11
num_x0s = 10

t0 = .52
rU = 2*t0
ra = 2

parameters = "-{}threadsperx0-{}numx0s-{}rU-{}ra".format(threads_per_x0, num_x0s, rU, ra)

target_Us = np.load('./GlobalMinimizeData/target_Us'+parameters+'.npy')
target_Us = np.reshape(target_Us, (5, 5))
target_as = np.load('./GlobalMinimizeData/target_as'+parameters+'.npy')
target_as = np.reshape(target_as, (5, 5))
runtimes = np.load('./GlobalMinimizeData/runtimes'+parameters+'.npy')
tot_time = sum(runtimes) / 3600
runtimes = np.reshape(runtimes, (5, 5))
final_costs = np.load('./GlobalMinimizeData/final_costs'+parameters+'.npy')
final_costs = np.reshape(final_costs, (5, 5))
final_Us = np.load('./GlobalMinimizeData/final_Us'+parameters+'.npy')
final_Us = np.reshape(final_Us, (5, 5))
final_as = np.load('./GlobalMinimizeData/final_as'+parameters+'.npy')
final_as = np.reshape(final_as, (5, 5))
U_percent_error = 100 * abs(final_Us - target_Us) / target_Us
a_percent_error = 100 * abs(final_as - target_as) / target_as

avg_U_err = np.mean(U_percent_error)
avg_a_error = np.mean(a_percent_error)

U_correct = sum([1 for x in U_percent_error.flatten() if x <= 10])
a_correct = sum([1 for x in a_percent_error.flatten() if x <= 10])

print("U correct:", U_correct)
print("a correct:", a_correct)

print("Average U error:", avg_U_err)
print("Average a error:", avg_a_error)

