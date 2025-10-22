import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../out.out', sep=", ")
data_wall = pd.read_csv('wall.csv', sep=", ")

plt.plot(data['ox'], data['oy'], "o")
# same aspect ratio
plt.axis('equal')

naca12 = np.loadtxt('naca0012.dat')
plt.plot(naca12[0,:], naca12[1,:])
plt.plot(data['nearest x'], data['nearest y'], "x")
# plt.quiver(data['ox'], data['oy'], data['vec x'], data['vec y'], angles='xy', scale_units='xy', scale=1, color='r')
plt.plot(data['x mirror'], data['y mirror'], "x")
# plt.quiver(data['nearest x'], data['nearest y'], data['vec x'], data['vec y'], angles='xy', scale_units='xy', scale=1, color='g')
# plt.plot(data_wall['cx'], data_wall['cy'], "o")
plt.quiver(data_wall['cx'], data_wall['cy'], (data_wall['u_mirror']+data_wall['u_ghost'])/2, (data_wall['v_mirror']+data_wall['v_ghost'])/2)
plt.show()