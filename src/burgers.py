import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import weno2


def flux(val):
    return (val ** 2) / 2.0

def flux_deriv(val):
    return val

def max_flux_deriv(a, b):
    return np.maximum(np.abs(a), np.abs(b))

def initial_condition(x_center):
    u0 = 0.5 + np.sin(np.pi * x_center)

    return u0

def exact_solution(x, time):
    x = np.remainder(x, 2.0)
    for i in range(N):
        if x[i] >= 1.0:
            x[i] -= 2.0

    u_exact = np.zeros(N)

    for i in (range(N)):
        u_exact[i] = sco.newton(exact_func, 0.5, fprime=exact_func_prime, args=(x[i], time), maxiter=5000)

    return u_exact

def exact_func(u, x, t):
    return 0.5 + np.sin(np.pi * (x - u * t)) - u

def exact_func_prime(u, x, t):
    return -np.pi * t * np.cos(np.pi * (x - u * t)) - 1

# Number of cells.
N = 160
a = -1.0
b = 1.0
CHAR_SPEED = 1.0
CFL_NUMBER = 0.6
T = 0.3183

w = weno2.Weno2(a, b, N, flux, flux_deriv, max_flux_deriv, CHAR_SPEED, CFL_NUMBER)
x_center = w.get_x_center()
u0 = initial_condition(x_center)

solution = u0
solution = w.integrate(u0, T)

plt.plot(x_center, solution, 'o', label='WENO, $k = 2$')
plt.plot(x_center, exact_solution(x_center, T), label='Exact')
plt.legend(loc='best')
plt.ylim([-0.5, 1.5])
plt.xticks(np.linspace(-1, 1, 11, endpoint=True))
plt.show()
# plt.savefig('/home/dima/weno2_advection_N=' + str(N) + '_T=' + str(T) + '.eps')
