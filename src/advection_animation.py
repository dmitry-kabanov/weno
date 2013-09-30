import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


w = weno2.Weno2(a, b, N, T, flux, flux_deriv, max_flux_deriv, CHAR_SPEED)
x_center = w.get_x_center()
u0 = initial_condition(x_center)
solution = u0

t = 0.0
nFrames = 200
t_final = 6.0
dt = t_final / nFrames

fig = plt.figure()
ax = plt.axes(xlim=(-1, 1), ylim=(-0.1, 1.1))
line_numeric, line_exact = ax.plot([], [], [], [], lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

plt.xlabel(r'$x$')
plt.ylabel(r'$u$')
# initialization function: plot the background of each frame
def init():
    global x_center, u0
    line_numeric.set_data(x_center, u0)
    line_exact.set_data(x_center, exact_solution(x_center, 0))
    time_text.set_text('time = %s' % 0.0)
    return line_numeric, line_exact, time_text

# animation function.  This is called sequentially
def animate(i):
    global x_center, solution, w
    time = dt * i
    solution = w.integrate(solution, time)
    exact_soln = exact_solution(x_center - CHAR_SPEED * dt * i, time)
    line_numeric.set_data(x_center, solution)
    line_exact.set_data(x_center, exact_soln)
    time_text.set_text('time = %s' % time)
    return line_numeric, line_exact, time_text

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nFrames, interval=20, blit=True)

anim.save('advection.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
