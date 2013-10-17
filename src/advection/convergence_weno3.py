import numpy as np
import matplotlib.pyplot as plt
import advection as advec
import weno3
import bootstrap as bs

bs.bootstrap()

N = [80, 160, 320, 640, 1280, 2560]
a = -1.0
b = 1.0
T = 1.0

errorsList = []
dxList = []
for n in N:
    w = weno3.Weno3(a, b, n, advec.flux, advec.flux_deriv, advec.max_flux_deriv, advec.CHAR_SPEED)
    x_center = w.get_x_center()
    u0 = advec.initial_condition_sine(x_center)
    solution = w.integrate(u0, T)
    exact = advec.exact_solution_sine(x_center - advec.CHAR_SPEED * T, T, n)
    dx = (b - a) / n
    errorsList.append(dx * np.linalg.norm(exact - solution, 1))
    dxList.append(dx)


plt.figure(figsize=[10, 7.5])
plt.loglog(dxList, errorsList, '-o', label=r'$\|error\|_1$')
plt.loglog(dxList, [1e8* dx**5 for dx in dxList], '-s', label=r'1e8$\cdot \Delta x^5$')
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$L_1$-norm of error')
plt.legend(loc='upper left')
plt.show()
# plt.savefig(bs.params['images_path'] + '/advection_convergence_weno3.eps')

for i in range(0, len(errorsList)):
    print N[i], errorsList[i]

