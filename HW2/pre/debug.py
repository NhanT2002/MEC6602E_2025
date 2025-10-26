import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for i in range(1, 500, 50):
#     data_wall = pd.read_csv(f'../debug/wall_{i}.csv', sep=", ")
#     plt.figure()
#     # same aspect ratio
#     plt.axis('equal')

#     naca12 = np.loadtxt('naca0012.dat')
#     plt.plot(naca12[0,:], naca12[1,:])
#     plt.quiver(data_wall['cx'], data_wall['cy'], data_wall['u'], data_wall['v'])
#     plt.xlim(0.6, 0.8)
#     plt.show()

data_wall = pd.read_csv(f'wall.csv', sep=", ")
plt.figure()
# same aspect ratio
plt.axis('equal')

naca12 = np.loadtxt('naca0012.dat')
plt.plot(naca12[0,:], naca12[1,:])
plt.quiver(data_wall['cx'], data_wall['cy'], data_wall['u_ib'], data_wall['v_ib'])
plt.xlim(0.1, 1.0)
plt.show()