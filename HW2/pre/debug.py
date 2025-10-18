import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../out.out', sep=" ", header=None)
x = data[1]
y = data[2]
nx = data[3]
ny = data[4]

plt.figure()
plt.quiver(x, y, nx, ny)
# same aspect ratio
plt.axis('equal')

data_naca = pd.read_csv('../naca.dat', sep=" ", header=None)
x_naca = data_naca.iloc[0, :].values
y_naca = data_naca.iloc[1, :].values
plt.plot(x_naca, y_naca, 'r-')

# zoom in
plt.xlim(0.8, 1.1)