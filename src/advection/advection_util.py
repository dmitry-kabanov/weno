import numpy as np
import matplotlib.pyplot as plt
import bootstrap as bs
import advection as advec

def plot_advection_solution(x, solution, ex_solution, T, N, weno_order):
    plt.plot(x, solution, 'o', label=r'WENO, $k = ' + str(weno_order) + r'$')
    plt.plot(x, ex_solution, label='Exact')
    plt.legend(loc='best')
    plt.ylim([-0.1, 1.1])
    plt.xticks(np.linspace(-1, 1, 11, endpoint=True))
    title = 'Time = ' + str(T) + ', ncells = ' + str(N)
    title += r', $\nu$ = ' + str(advec.CFL_NUMBER)
    plt.title(title)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    # plt.show()
    filename = bs.params['images_path'] + '/advection_weno' + str(weno_order)
    filename += '_T=' + str(T) + '_N=' + str(N) + '.eps'
    plt.savefig(filename)

