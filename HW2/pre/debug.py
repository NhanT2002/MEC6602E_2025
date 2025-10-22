import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_wall = pd.read_csv('wall.csv', sep=", ")
# same aspect ratio
plt.axis('equal')

naca12 = np.loadtxt('naca0012.dat')
plt.plot(naca12[0,:], naca12[1,:])
plt.quiver(data_wall['cx'], data_wall['cy'], data_wall['u'], data_wall['v'])
plt.xlim(0.0, 0.1)
plt.show()