import matplotlib.pyplot as plt
import matplotlib.animation as animation
from advection import advection as advec
import weno2
import bootstrap as bs

bs.bootstrap()

N = 320
a = -1.0
b = 1.0

w = weno2.Weno2(a, b, N, advec.flux, advec.flux_deriv, advec.max_flux_deriv, advec.CHAR_SPEED)
x_center = w.get_x_center()
u0 = advec.initial_condition(x_center)
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
    line_exact.set_data(x_center, advec.exact_solution(x_center, 0, N))
    time_text.set_text('time = %s' % 0.0)
    return line_numeric, line_exact, time_text

# animation function.  This is called sequentially
def animate(i):
    global x_center, solution, w
    time = dt * i
    solution = w.integrate(solution, time)
    exact_soln = advec.exact_solution(x_center - advec.CHAR_SPEED * dt * i, time, N)
    line_numeric.set_data(x_center, solution)
    line_exact.set_data(x_center, exact_soln)
    time_text.set_text('time = %s' % time)
    return line_numeric, line_exact, time_text

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nFrames, interval=20, blit=True)

anim.save(bs.params['videos_path'] + '/advection_weno3.mp4', fps=30, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
