import numpy as np

t = .52
U_lower = 0
U_upper = 10 * t
a_lower = 0
a_upper = 10

rU = .4 * t
ra = .4

test_arr = np.array([0,1])

for best_U in [.1, 3, 5.1]:
    # [U_lower, U_lower + rU*t)
    if best_U - U_lower < rU:
        print(rU * test_arr)

    # [U_upper - rU * t, U_upper)
    elif U_upper - best_U < rU:
        print(rU * test_arr + (U_upper - rU))

    # [U-rU/2, U+rU/2)
    else:
        print((rU * (test_arr - .5)) + best_U)

for best_a in [.2, 5, 9.8]:
    # [a_lower , a_lower + ra)
    if best_a - a_lower < ra:
        print(ra * test_arr)

    # [a_upper - ra, a_upper)
    elif a_upper - best_a < ra:
        print(ra * test_arr + (a_upper - ra))

    # [a-ra/2,a+ra/2)
    else:
        print((ra * (test_arr - .5)) + best_a)
