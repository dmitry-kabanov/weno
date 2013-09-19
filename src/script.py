import numpy as np

N = 100
x_boundary = np.linspace(a, b, N + 1)
x_center = np.zeros(N+1)

for i in range(1, N+2):
    x_center[i] = (x_boundary[i-1] + x_boundary[i]) / 2.0

u = np.zeros(N+1)

for i in range(1, N+2):
    if (x_center[i] >= -0.2 and x_center[i] <= 0.2):
        u[i] = 1.0
    else:
        u[i] = 0.0

dx = (b - a) / (N + 0.0)
CFL_NUMBER = 0.5
CHAR_SPEED = 1.0
dt = CFL_NUMBER * dx / CHAR_SPEED
T = 0.5
t = 0.0

while (t < T):

