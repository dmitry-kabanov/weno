import numpy as np
import matplotlib.pyplot as plt
import weno3
import advection as advec

def plot_advection_solution():
    plt.plot(x_center, solution, 'o', label='WENO, $k = 3$')
    plt.plot(x_center, advec.exact_solution(x_center - advec.CHAR_SPEED * T, T, N), label='Exact')
    plt.legend(loc='best')
    plt.ylim([-0.1, 1.1])
    plt.xticks(np.linspace(-1, 1, 11, endpoint=True))
    plt.gca().text(0.02, 0.95, 'time = ' + str(T), transform=plt.gca().transAxes)
    plt.show()
    # plt.savefig('/home/dima/weno2_advection_N=' + str(N) + '_T=' + str(T) + '.eps')


N = 160
a = -1.0
b = 1.0
T = 4.0

w = weno3.Weno3(a, b, N, advec.flux, advec.flux_deriv, advec.max_flux_deriv, advec.CHAR_SPEED, CFL_NUMBER)
x_center = w.get_x_center()
u0 = advec.initial_condition(x_center)

solution = w.integrate(u0, T)
plot_advection_solution()


