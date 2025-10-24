import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for i in range(1, 25) : 
    data_wall = pd.read_csv(f'../wall_{i}.csv', sep=", ")
    plt.figure(figsize=(20,20))
    # same aspect ratio
    plt.axis('equal')

    naca12 = np.loadtxt('naca0012.dat')
    plt.plot(naca12[0,:], naca12[1,:])
    plt.quiver(data_wall['cx'], data_wall['cy'], data_wall['u'], data_wall['v'])
    plt.xlim(0.0, 0.1)
    plt.show()