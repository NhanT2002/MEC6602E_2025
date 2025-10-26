import numpy as np
import matplotlib.pyplot as plt
import h5py

def data(filename) :
    # Read CGNS file
    with h5py.File(filename, 'r') as f:
        X = f['Base/WallZone/WallCellData/CoordinateX/ data'][:].squeeze()
        Y = f['Base/WallZone/WallCellData/CoordinateY/ data'][:].squeeze()

        rho = f['Base/WallZone/WallCellData/rho/ data'][:].squeeze()
        u = f['Base/WallZone/WallCellData/u/ data'][:].squeeze()
        v = f['Base/WallZone/WallCellData/v/ data'][:].squeeze()
        p = f['Base/WallZone/WallCellData/p/ data'][:].squeeze()
        E = f['Base/WallZone/WallCellData/E/ data'][:].squeeze()
        Cp = f['Base/WallZone/WallCellData/Cp/ data'][:].squeeze()

    return X, Y, rho, u, v, p, E, Cp

X, Y, rho, u, v, p, E, Cp = data('../output/output_99.cgns')
plt.figure()
plt.xlim(0.0, 1.0)
plt.plot(X, Cp, "o")
plt.grid()
# inverse y-axis
plt.gca().invert_yaxis()


# plt.quiver(X, Y, u, v)
# plt.xlim(0, 0.1)