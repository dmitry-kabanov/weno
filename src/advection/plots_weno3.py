import advection as advec
import advection_util as au
import weno3
import bootstrap as bs

bs.bootstrap()

N = 640
T = 0.5
WENO_ORDER = 3

w = weno3.Weno3(advec.a, advec.b, N, advec.flux, advec.flux_deriv, advec.max_flux_deriv, advec.CHAR_SPEED, advec.CFL_NUMBER)
x_center = w.get_x_center()
u0 = advec.initial_condition_square_wave(x_center)

solution = w.integrate(u0, T)
exact_solution = advec.exact_solution_square_wave(x_center - advec.CHAR_SPEED * T, T, N)
au.plot_advection_solution(x_center, solution, exact_solution, T, N, WENO_ORDER)


