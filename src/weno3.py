import numpy as np

class Weno3:
    def __init__(self, left_boundary, right_boundary, ncells, flux_callback, flux_deriv_callback, max_flux_deriv_callback, char_speed, cfl_number=0.8, eps=1.0e-6):
        """

        :rtype : Weno3
        """
        a = left_boundary
        b = right_boundary
        self.N = ncells
        self.dx = (b - a) / (self.N + 0.0)
        self.CFL_NUMBER = cfl_number
        self.CHAR_SPEED = char_speed
        self.t = 0.0
        ORDER_OF_SCHEME = 3
        self.EPS = eps
        self.D0 = 3.0 / 10.0
        self.D1 = 6.0 / 10.0
        self.D2 = 1.0 / 10.0
        self.flux = flux_callback
        self.flux_deriv = flux_deriv_callback
        self.max_flux_deriv = max_flux_deriv_callback
        self.x_boundary = np.linspace(a, b, self.N + 1)
        self.x_center = np.zeros(self.N)

        for i in range(0, self.N):
            self.x_center[i] = (self.x_boundary[i] + self.x_boundary[i + 1]) / 2.0

        self.u_right_boundary_approx = np.zeros((ORDER_OF_SCHEME, self.N))
        self.u_left_boundary_approx = np.zeros((ORDER_OF_SCHEME, self.N))
        self.u_right_boundary = np.zeros(self.N)
        self.u_left_boundary = np.zeros(self.N)
        self.beta0 = np.zeros(self.N)
        self.beta1 = np.zeros(self.N)
        self.beta2 = np.zeros(self.N)
        self.alpha0 = np.zeros(self.N)
        self.alpha1 = np.zeros(self.N)
        self.alpha2 = np.zeros(self.N)
        self.alpha0_left = np.zeros(self.N)
        self.alpha1_left = np.zeros(self.N)
        self.alpha2_left = np.zeros(self.N)
        self.sum_alpha_left = np.zeros(self.N)
        self.omega0_left = np.zeros(self.N)
        self.omega1_left = np.zeros(self.N)
        self.omega2_left = np.zeros(self.N)
        self.fFlux = np.zeros(self.N + 1)
        self.rhsValues = np.zeros(self.N)
        self.u_multistage = np.zeros((3, self.N))


    def integrate(self, u0, time_final):
        self.dt = self.CFL_NUMBER * self.dx / self.CHAR_SPEED
        self.T = time_final
        self.u_multistage[0] = u0

        while self.t < self.T:
            if self.t + self.dt > self.T:
                self.dt = self.T - self.t

            self.t += self.dt
            self.u_multistage[1] = self.u_multistage[0] + self.dt * self.rhs(self.u_multistage[0])
            self.u_multistage[2] = (3 * self.u_multistage[0] + self.u_multistage[1] + self.dt * self.rhs(self.u_multistage[1])) / 4.0
            self.u_multistage[0] = (self.u_multistage[0] + 2.0 * self.u_multistage[2] + 2.0 * self.dt * self.rhs(self.u_multistage[2])) / 3.0

        return self.u_multistage[0]


    def rhs(self, u):
        # WENO Reconstruction
        # Approximations for inner cells 0 < i < N-1.
        self.u_right_boundary_approx[0][2:-2] = 1.0 / 3.0 * u[2:-2] + 5.0 / 6.0 * u[3:-1] - 1.0 / 6.0 * u[4:]
        self.u_right_boundary_approx[1][2:-2] = -1.0 / 6.0 * u[1:-3] + 5.0 / 6.0 * u[2:-2] + 1.0 / 3.0 * u[3:-1]
        self.u_right_boundary_approx[2][2:-2] = 1.0 / 3.0 * u[0:-4] - 7.0 / 6.0 * u[1:-3] + 11.0 / 6.0 * u[2:-2]
        self.u_left_boundary_approx[0][2:-2] = 11.0 / 6.0 * u[2:-2] - 7.0 / 6.0 * u[3:-1] + 1.0 / 3.0 * u[4:]
        self.u_left_boundary_approx[1][2:-2] = 1.0 / 3.0 * u[1:-3] + 5.0 / 6.0 * u[2:-2] - 1.0 / 6.0 * u[3:-1]
        self.u_left_boundary_approx[2][2:-2] = -1.0 / 6.0 * u[0:-4] + 5.0 / 6.0 * u[1:-3] + 1.0 / 3.0 * u[2:-2]

        # Approximations for cell i = 0 (the leftmost cell).
        self.u_right_boundary_approx[0][0] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
        self.u_right_boundary_approx[1][0] = -1.0 / 6.0 * u[-1] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
        self.u_right_boundary_approx[2][0] = 1.0 / 3.0 * u[-2] - 7.0 / 6.0 * u[-1] + 11.0 / 6.0 * u[0]
        self.u_left_boundary_approx[0][0] = 11.0 / 6.0 * u[0] - 7.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
        self.u_left_boundary_approx[1][0] = 1.0 / 3.0 * u[-1] + 5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
        self.u_left_boundary_approx[2][0] = -1.0 / 6.0 * u[-2] + 5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]

        # Approximations for cell i = 1.
        self.u_right_boundary_approx[0][1] = 1.0 / 3.0 * u[1] + 5.0 / 6.0 * u[2] - 1.0 / 6.0 * u[3]
        self.u_right_boundary_approx[1][1] = -1.0 / 6.0 * u[0] + 5.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
        self.u_right_boundary_approx[2][1] = 1.0 / 3.0 * u[-1] - 7.0 / 6.0 * u[0] + 11.0 / 6.0 * u[1]
        self.u_left_boundary_approx[0][1] = 11.0 / 6.0 * u[1] - 7.0 / 6.0 * u[2] + 1.0 / 3.0 * u[3]
        self.u_left_boundary_approx[1][1] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
        self.u_left_boundary_approx[2][1] = -1.0 / 6.0 * u[-1] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]

        # Approximations for cell i = N-2.
        self.u_right_boundary_approx[0][-2] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[0]
        self.u_right_boundary_approx[1][-2] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]
        self.u_right_boundary_approx[2][-2] = 1.0 / 3.0 * u[-4] - 7.0 / 6.0 * u[-3] + 11.0 / 6.0 * u[-2]
        self.u_left_boundary_approx[0][-2] = 11.0 / 6.0 * u[-2] - 7.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
        self.u_left_boundary_approx[1][-2] = 1.0 / 3.0 * u[-3] + 5.0 / 6.0 * u[-2] - 1.0 / 6.0 * u[-1]
        self.u_left_boundary_approx[2][-2] = -1.0 / 6.0 * u[-4] + 5.0 / 6.0 * u[-3] + 1.0 / 3.0 * u[-2]

        # Approximations for cell i = N-1 (the rightmost cell).
        self.u_right_boundary_approx[0][-1] = 1.0 / 3.0 * u[-1] + 5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
        self.u_right_boundary_approx[1][-1] = -1.0 / 6.0 * u[-2] + 5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
        self.u_right_boundary_approx[2][-1] = 1.0 / 3.0 * u[-3] - 7.0 / 6.0 * u[-2] + 11.0 / 6.0 * u[-1]
        self.u_left_boundary_approx[0][-1] = 11.0 / 6.0 * u[-1] - 7.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
        self.u_left_boundary_approx[1][-1] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[0]
        self.u_left_boundary_approx[2][-1] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]

        self.beta0[2:-2] = 13.0 / 12.0 * (u[2:-2] - 2 * u[3:-1] + u[4:]) ** 2 + \
                           1.0 / 4.0 * (3*u[2:-2] - 4.0 * u[3:-1] + u[4:]) ** 2
        self.beta1[2:-2] = 13.0 / 12.0 * (u[1:-3] - 2 * u[2:-2] + u[3:-1]) ** 2 + \
                           1.0 / 4.0 * (u[1:-3] - u[3:-1]) ** 2
        self.beta2[2:-2] = 13.0 / 12.0 * (u[0:-4] - 2 * u[1:-3] + u[2:-2]) ** 2 + \
                           1.0 / 4.0 * (u[0:-4] - 4.0 * u[1:-3] + 3 * u[2:-2]) ** 2

        self.beta0[0] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
                           1.0 / 4.0 * (3*u[0] - 4.0 * u[1] + u[2]) ** 2
        self.beta1[0] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
                           1.0 / 4.0 * (u[-1] - u[1]) ** 2
        self.beta2[0] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
                           1.0 / 4.0 * (u[-2] - 4.0 * u[-1] + 3 * u[0]) ** 2

        self.beta0[1] = 13.0 / 12.0 * (u[1] - 2 * u[2] + u[3]) ** 2 + \
                        1.0 / 4.0 * (3*u[1] - 4.0 * u[2] + u[3]) ** 2
        self.beta1[1] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
                        1.0 / 4.0 * (u[0] - u[2]) ** 2
        self.beta2[1] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
                        1.0 / 4.0 * (u[-1] - 4.0 * u[0] + 3 * u[1]) ** 2

        self.beta0[-2] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
                        1.0 / 4.0 * (3*u[-2] - 4.0 * u[-1] + u[0]) ** 2
        self.beta1[-2] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
                        1.0 / 4.0 * (u[-3] - u[-1]) ** 2
        self.beta2[-2] = 13.0 / 12.0 * (u[-4] - 2 * u[-3] + u[-2]) ** 2 + \
                        1.0 / 4.0 * (u[-4] - 4.0 * u[-3] + 3 * u[-2]) ** 2

        self.beta0[-1] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
                        1.0 / 4.0 * (3*u[-1] - 4.0 * u[0] + u[1]) ** 2
        self.beta1[-1] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
                        1.0 / 4.0 * (u[-2] - u[0]) ** 2
        self.beta2[-1] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
                        1.0 / 4.0 * (u[-3] - 4.0 * u[-2] + 3 * u[-1]) ** 2
        self.alpha0 = self.D0 / ((self.EPS + self.beta0) ** 2)
        self.alpha1 = self.D1 / ((self.EPS + self.beta1) ** 2)
        self.alpha2 = self.D2 / ((self.EPS + self.beta2) ** 2)
        self.alpha0_left = self.D2 / ((self.EPS + self.beta0) ** 2)
        self.alpha1_left = self.D1 / ((self.EPS + self.beta1) ** 2)
        self.alpha2_left = self.D0 / ((self.EPS + self.beta2) ** 2)
        self.sum_alpha = self.alpha0 + self.alpha1 + self.alpha2
        self.sum_alpha_left = self.alpha0_left + self.alpha1_left + self.alpha2_left
        self.omega0 = self.alpha0 / self.sum_alpha
        self.omega1 = self.alpha1 / self.sum_alpha
        self.omega2 = self.alpha2 / self.sum_alpha
        self.omega0_left = self.alpha0_left / self.sum_alpha_left
        self.omega1_left = self.alpha1_left / self.sum_alpha_left
        self.omega2_left = self.alpha2_left / self.sum_alpha_left
        self.u_right_boundary = self.omega0 * self.u_right_boundary_approx[0][:] + \
                           self.omega1 * self.u_right_boundary_approx[1][:] + \
                           self.omega2 * self.u_right_boundary_approx[2][:]
        self.u_left_boundary = self.omega0_left * self.u_left_boundary_approx[0][:] + \
                          self.omega1_left * self.u_left_boundary_approx[1][:] + \
                          self.omega2_left * self.u_left_boundary_approx[2][:]

        # Numerical flux calculation.
        self.fFlux[1:-1] = self.numflux(self.u_right_boundary[0:-1], self.u_left_boundary[1:])
        self.fFlux[0] = self.numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])
        self.fFlux[self.N] = self.numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])

        # Right hand side calculation.
        rhsValues = self.fFlux[1:] - self.fFlux[0:-1]
        rhsValues = -rhsValues / self.dx

        return rhsValues


    def numflux(self, a, b):
        """
        Return Lax-Friedrichs numerical flux.
        """
        maxval = self.max_flux_deriv(a, b)

        return 0.5 * (self.flux(a) + self.flux(b) - maxval * (b - a))


    def get_x_center(self):
        return self.x_center
