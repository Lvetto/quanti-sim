import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from simulation import *

# simulation parameters
grid_size = 250
x0 = 50
y0 = 125
kx = -50
ky = 0
sigma = 5
m = 1

# generate potetial
x = np.linspace(0, grid_size, grid_size)
y = np.linspace(0, grid_size, grid_size)
xx, yy = np.meshgrid(x, y)
V = np.zeros((grid_size, grid_size))
V = 0.000 * ((500-xx)**2+(500-yy)**2)**(1/2)
#V[(xx < 200) & (xx < 500)] = 0

# used to apply boundary condition (psi=0 on the edges)
boundary_mat = np.pad(np.ones((grid_size-2, grid_size-2)), pad_width=1, mode="constant", constant_values=0)

psi = wave_packet(x0, y0, kx, ky, sigma, grid_size)

# create heatmaps
fig, axs = plt.subplots(2, 2)   # abs(psi)**2, potential, real and imaginary parts of psi on the bottom

# set initial data and titles to the subplots
axs[0, 0].title.set_text('|psi|^2')
cax1 = axs[0, 0].imshow(np.abs(psi)**2, cmap='cool')
plt.colorbar(cax1, ax=axs[0, 0])

axs[0, 1].title.set_text('V')
cax2 = axs[0, 1].imshow(V, cmap='cool')
plt.colorbar(cax2, ax=axs[0, 1])

axs[1, 0].title.set_text('Real(psi)')
cax3 = axs[1, 0].imshow(np.real(psi), cmap='cool')
plt.colorbar(cax3, ax=axs[1, 0])

axs[1, 1].title.set_text('Imag(psi)')
cax4 = axs[1, 1].imshow(np.imag(psi), cmap='cool')
plt.colorbar(cax4, ax=axs[1, 1])


# update animation
def animate(i):
    global psi

    psi = integrate_rk4(psi, m, V, 0.5) * boundary_mat

    d1 = np.abs(psi)**2
    cax1.set_data(d1)
    cax1.set_clim(np.min(d1), np.max(d1))


    cax2.set_data(np.real(V))

    d2 = np.real(psi)
    cax3.set_data(np.real(psi))
    cax3.set_clim(np.min(d2), np.max(d2))

    d3 = np.imag(psi)
    cax4.set_data(d3)
    cax4.set_clim(np.min(d3), np.max(d3))



    return cax1, cax2, cax3, cax4,

# run animation
anim = animation.FuncAnimation(fig, animate, frames=100000, interval=1, blit=True)

# Show graphs
plt.show()
