import numpy as np
import matplotlib.pyplot as plt
import burgers as b
import weno2

def plot_burgers_solution():
    plt.plot(x_center, solution, 'o', label='WENO, $k = 2$')
    if T <= time_breakage:
        plt.plot(x_center, b.exact_solution(x_center, T, ncells), label='Exact')
    plt.legend(loc='best')
    plt.ylim([-0.55, 1.55])
    plt.xticks(np.linspace(-1, 1, 11, endpoint=True))
    plt.show()
    # plt.savefig('/home/dima/weno2_advection_N=' + str(N) + '_T=' + str(T) + '.eps')

ncells = 320
left_boundary = -1.0
right_boundary = 1.0
CHAR_SPEED = 1.0
CFL_NUMBER = 0.6
time_breakage = 0.3183
T = 0.2

w = weno2.Weno2(left_boundary, right_boundary, ncells, b.flux, b.flux_deriv, b.max_flux_deriv, CHAR_SPEED, CFL_NUMBER)
x_center = w.get_x_center()
u0 = b.initial_condition(x_center)

solution = w.integrate(u0, T)
plot_burgers_solution()
