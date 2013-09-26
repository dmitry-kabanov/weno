import numpy as np
import matplotlib.pyplot as plt

def rhs(u):
    # WENO Reconstruction
    for i in range(0, N):
        if i > 0 and i < N - 1:
            u_right_boundary_approx[i][0] = 0.5 * u[i] + 0.5 * u[i + 1]
            u_right_boundary_approx[i][1] = -0.5 * u[i - 1] + 1.5 * u[i]
            u_left_boundary_approx[i][0] = 1.5 * u[i] - 0.5 * u[i + 1]
            u_left_boundary_approx[i][1] = 0.5 * u[i - 1] + 0.5 * u[i]

        if i == 0:
            u_right_boundary_approx[0][0] = 0.5 * u[0] + 0.5 * u[1]
            u_right_boundary_approx[0][1] = -0.5 * u[N - 1] + 1.5 * u[0]
            u_left_boundary_approx[0][0] = 1.5 * u[0] - 0.5 * u[1]
            u_left_boundary_approx[0][1] = 0.5 * u[N - 1] + 0.5 * u[0]

        if i == (N - 1):
            u_right_boundary_approx[N - 1][0] = 0.5 * u[N - 1] + 0.5 * u[0]
            u_right_boundary_approx[N - 1][1] = -0.5 * u[N - 2] + 1.5 * u[N - 1]
            u_left_boundary_approx[N - 1][0] = 1.5 * u[N - 1] - 0.5 * u[0]
            u_left_boundary_approx[N - 1][1] = 0.5 * u[N - 2] + 0.5 * u[N - 1]
    for i in range(0, N):
        if 0 < i < (N - 1):
            beta0 = (u[i + 1] - u[i]) ** 2
            beta1 = (u[i] - u[i - 1]) ** 2

        if i == 0:
            beta0 = (u[1] - u[0]) ** 2
            beta1 = (u[0] - u[N - 1]) ** 2

        if i == (N - 1):
            beta0 = (u[0] - u[N - 1]) ** 2
            beta1 = (u[N - 1] - u[N - 2]) ** 2

        alpha0 = D0 / ((EPS + beta0) ** 2)
        alpha1 = D1 / ((EPS + beta1) ** 2)

        sum_alpha = alpha0 + alpha1

        omega0 = alpha0 / sum_alpha
        omega1 = alpha1 / sum_alpha

        u_right_boundary[i] = omega0 * u_right_boundary_approx[i][0] + \
                              omega1 * u_right_boundary_approx[i][1]

        u_left_boundary[i] = omega0 * u_left_boundary_approx[i][0] + \
                             omega1 * u_left_boundary_approx[i][1]


    # Numerical flux calculation.
    fFlux[1:-1] = numflux(u_right_boundary[0:-1], u_left_boundary[1:-1])
    fFlux[0] = numflux(u_right_boundary[N - 1], u_left_boundary[0])
    fFlux[N] = numflux(u_right_boundary[N - 1], u_left_boundary[0])

    # Right hand side calculation.
    rhs = np.zeros(N)
    rhs[:] = fFlux[1:] - fFlux[0:-1]
    rhs = -rhs / dx

    return rhs


def numflux(a, b):
    if flux_deriv(a) > 0:
        return flux(a)
    elif flux_deriv(b) < 0:
        return flux(b)
    else:
        raise Exception("Numerical flux cannot deal with sonic points.")

def flux(val):
    return val

def flux_deriv(val):
    return CHAR_SPEED


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
dx = (b - a) / (N + 0.0)
CFL_NUMBER = 0.8
CHAR_SPEED = 1.0
dt = CFL_NUMBER * dx / CHAR_SPEED
T = 4.0
t = 0.0

ORDER_OF_SCHEME = 2

EPS = 1.0e-6
D0 = 2.0 / 3.0
D1 = 1.0 / 3.0
x_boundary = np.linspace(a, b, N + 1)
x_center = np.zeros(N)

for i in range(0, N):
    x_center[i] = (x_boundary[i] + x_boundary[i + 1]) / 2.0

u0 = np.zeros(N)

for i in range(0, N):
    if -0.2 <= x_center[i] <= 0.2:
        u0[i] = 1.0
    else:
        u0[i] = 0.0


u_right_boundary_approx = np.zeros((N, ORDER_OF_SCHEME))
u_left_boundary_approx = np.zeros((N, ORDER_OF_SCHEME))
u_right_boundary = np.zeros(N)
u_left_boundary = np.zeros(N)
fFlux = np.zeros(N+1)

u_multistage = np.zeros((3, N))
u_multistage[0] = u0

while t < T:
    u_multistage[1] = u_multistage[0] + dt * rhs(u_multistage[0])
    u_multistage[2][:] = (3 * u_multistage[0] + u_multistage[1] + dt * rhs(u_multistage[1])) / 4.0
    u_multistage[0] = (u_multistage[0] + 2.0 * u_multistage[2] + 2.0 * dt * rhs(u_multistage[2])) / 3.0
    t += dt


plt.plot(x_center, u_multistage[0], 'o', label='WENO, $k = 2$')
plt.plot(x_center, exact_solution(x_center - CHAR_SPEED * t, t), label='Exact')
plt.legend(loc='best')
plt.ylim([-0.1, 1.1])
plt.xticks(np.linspace(-1, 1, 11, endpoint=True))
plt.show()
# plt.savefig('/home/dima/weno2_advection_T=' + str(T) + '.eps')
