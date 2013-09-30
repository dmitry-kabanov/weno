import numpy as np
import matplotlib.pyplot as plt
import weno2


def flux(val):
    return val

def flux_deriv(val):
    return CHAR_SPEED

def max_flux_deriv(a, b):
    return CHAR_SPEED

def initial_condition(x_center):
    u0 = np.zeros(N)

    for i in range(0, N):
        if -0.2 <= x_center[i] <= 0.2:
            u0[i] = 1.0
        else:
            u0[i] = 0.0

    return u0

def exact_solution(x, time):
    x = np.remainder(x, 2.0)
    for i in range(N):
        if x[i] >= 1.0:
            x[i] -= 2.0

    u_exact = np.zeros(N)

    for i in (range(N)):
        if -0.2 <= x[i] <= 0.2:
            u_exact[i] = 1.0

    return u_exact


# Number of cells.
N = 160
a = -1.0
b = 1.0
CHAR_SPEED = 1.0
T = 4.0

w = weno2.Weno2(a, b, N, flux, flux_deriv, max_flux_deriv, CHAR_SPEED)
x_center = w.get_x_center()
u0 = initial_condition(x_center)

solution = w.integrate(u0, T)

plt.plot(x_center, solution, 'o', label='WENO, $k = 2$')
plt.plot(x_center, exact_solution(x_center - CHAR_SPEED * T, T), label='Exact')
plt.legend(loc='best')
plt.ylim([-0.1, 1.1])
plt.xticks(np.linspace(-1, 1, 11, endpoint=True))
plt.show()
# plt.savefig('/home/dima/weno2_advection_N=' + str(N) + '_T=' + str(T) + '.eps')
