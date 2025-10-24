import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_wall = pd.read_csv('wall.csv', sep=", ")
# same aspect ratio
plt.axis('equal')

naca12 = np.loadtxt('naca0012.dat')
plt.plot(naca12[0,:], naca12[1,:])
plt.quiver(data_wall['cx'], data_wall['cy'], data_wall['u_BI'], data_wall['v_BI'])
plt.quiver(data_wall['x_mirror'], data_wall['y_mirror'], data_wall['u_mirror'], data_wall['v_mirror'])
plt.xlim(0.5, 0.6)
plt.show()