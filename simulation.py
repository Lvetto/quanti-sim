import numpy as np
import math
from scipy.ndimage import convolve

# compute the laplacian on a 2d grid by convolluting with a kernel representing a first order approximation of the laplacian operator
def laplacian_2d(grid, dx=1):

    # kernel representing the laplacian operator
    kernel = np.array([     [0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]]) / (dx**2)
    
    # apply the kernel. The grid is treated as padded with (complex) zeros
    return convolve(grid, kernel, mode="constant", cval=0j)

# generate a gaussian wave packet given its parameters
def wave_packet(x0, y0, kx, ky, sigma, grid_size):

    # creates the grid and objects representing rows and columns
    x = np.linspace(0, grid_size, grid_size)
    y = np.linspace(0, grid_size, grid_size)
    X, Y = np.meshgrid(x, y)

    # apply the wave packet equation to every point in the grid
    psi_0 = math.e**(-((X-x0)**2 + (Y-y0)**2)/(4*sigma)) * math.e**(1j*(kx*X+ky*Y))

    return psi_0

# approximates the rate of change of psi, given a potential, mass and time step
def compute_derivative(psi, m, V):
    # delta = 1\(i) * (delta_1 (\propto laplacian) + delta_2 (V*psi))

    delta_1 = (-1/(2*m)) * laplacian_2d(psi)
    delta_2 = V * psi

    return (1/1j) * delta_1 + delta_2

# numerical integration with fourth order runge-kutta method
def integrate_rk4(psi, m, V, dt):
    # compute time-independent runge-kutta coefficients
    k1 = dt * compute_derivative(psi, m, V)
    k2 = dt * compute_derivative(psi + k1/2, m, V)
    k3 = dt * compute_derivative(psi + k2/2, m, V)
    k4 = dt * compute_derivative(psi + k3, m ,V)

    # applies correction to psi(t) given by weighted average of the coefficients
    return psi + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
