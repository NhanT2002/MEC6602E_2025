import numpy as np
import pandas as pd

def read_PLOT3D_mesh(file_name):
    with open(file_name, 'r') as f:
        # Read all lines from the file
        data = f.readlines()
        
        # Extract the grid dimensions
        nx, ny = map(int, data[0].split())
        
        # Calculate the total number of points
        total_points = nx * ny
        
        # Initialize arrays for coordinates
        x = np.zeros((total_points))
        y = np.zeros((total_points))
        
        # Extract the coordinates (assuming they are in a single column, alternating X and Y)
        for i in range(total_points):
            x[i] = float(data[i+1])
            y[i] = float(data[i+1+total_points])
        
        x = x.reshape((ny, nx))
        y = y.reshape((ny, nx))
            
    return x, y

def read_plot3d_2d(solution_filename):
    """
    Read 2D PLOT3D formatted solution file (q file).

    Parameters:
        solution_filename: Name of the solution file to read.

    Returns:
        ni, nj: Grid dimensions.
        mach, alpha, reyn, time: Freestream conditions.
        q: 3D NumPy array (nj, ni, 4) representing flow variables (density, x-momentum, y-momentum, energy).
    """
    with open(solution_filename, 'r') as solution_file:
        # Read grid dimensions
        ni, nj = map(int, solution_file.readline().split())
        
        # Read freestream conditions
        mach, alpha, reyn, time = map(float, solution_file.readline().split())

        # Initialize the q array (nj, ni, 4)
        q = np.zeros((nj, ni, 4))
        
        # Read flow variables
        for n in range(4):  # Iterate over the 4 variables (density, x-momentum, y-momentum, energy)
            for j in range(nj):
                for i in range(ni):  # Read in the reversed order: i first, then j
                    q[j, i, n] = float(solution_file.readline())

    return ni, nj, mach, alpha, reyn, time, q

def conservative_variable_from_W(W, gamma=1.4):
    variable = W/W[0]
    rho = W[0]
    u = variable[1]
    v = variable[2]
    E = variable[3]
    p = (gamma-1)*rho*(E-(u**2+v**2)/2)
    T = p/(rho*287)
    
    return rho, u, v, E, T, p

class cell :
    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4, rho, u, v, E) :
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4
        
        self.rho = rho
        self.u = u
        self.v = v
        self.E = E
        
        
        self.OMEGA = 0.5*((x1-x3)*(y2-y4) + (x4-x2)*(y1-y3))
        
        self.s1 = np.array([self.y2-self.y1, self.x1-self.x2])
        self.s2 = np.array([self.y3-self.y2, self.x2-self.x3])
        self.s3 = np.array([self.y4-self.y3, self.x3-self.x4])
        self.s4 = np.array([self.y1-self.y4, self.x4-self.x1])
        
        self.Ds1 = np.linalg.norm(self.s1)
        self.Ds2 = np.linalg.norm(self.s2)
        self.Ds3 = np.linalg.norm(self.s3)
        self.Ds4 = np.linalg.norm(self.s4)
        
        self.n1 = self.s1/self.Ds1
        self.n2 = self.s2/self.Ds2
        self.n3 = self.s3/self.Ds3
        self.n4 = self.s4/self.Ds4
        
        self.W = np.array([rho, rho*u, rho*v, rho*E])
        
        self.FcDS_1 = np.array(4)
        self.FcDS_2 = np.array(4)
        self.FcDS_3 = np.array(4)
        self.FcDS_4 = np.array(4)
        
        self.Lambda_1_I = 0.
        self.Lambda_1_J = 0.
        self.Lambda_2_I = 0.
        self.Lambda_2_J = 0.
        self.Lambda_3_I = 0.
        self.Lambda_3_J = 0.
        self.Lambda_4_I = 0.
        self.Lambda_4_J = 0.
        
        self.Lambda_1_S = 0.
        self.Lambda_2_S = 0.
        self.Lambda_3_S = 0.
        self.Lambda_4_S = 0.
        
        self.eps2_2, self.eps4_2 = 0., 0.
        self.eps2_3, self.eps4_3 = 0., 0.
        self.eps2_4, self.eps4_4 = 0., 0.
        self.eps2_1, self.eps4_1 = 0., 0.

        
        self.D_1 = 0.
        self.D_2 = 0.
        self.D_3 = 0.
        self.D_4 = 0.
        
        self.R = np.array(4)

def compute_coeff(x, y, q, Mach, alpha, T_inf, p_inf, chord=1.00893):
    a_inf = np.sqrt(1.4 * 287 * T_inf)  # Freestream speed of sound
    U_inf = Mach * a_inf  # Freestream velocity magnitude
    rho_inf = p_inf/(T_inf*287)
    ny, nx, n = q.shape
    
    q_airfoil = np.zeros((nx, 6))
    for i in range(nx) :
        q_airfoil[i] = conservative_variable_from_W(q[0, i, :])
    
    # Cells generation
    airfoil_cells = np.zeros((nx-1), dtype=object)
    for i in range(nx-1) :
        airfoil_cells[i] = cell(x[0, i], y[0, i], x[0, i+1], y[0, i+1], x[0+1, i+1], y[0+1, i+1], x[0+1, i], y[0+1, i], 0., 0., 0., 0.)
    
    cp_airfoil = (q_airfoil[:, 5] - p_inf)/(0.5*rho_inf*U_inf**2)
    
    Fx = 0
    Fy = 0
    M = 0
    x_ref = chord/4
    y_ref = 0
    for i in range(nx - 1):
        p_mid = 0.5 * (q_airfoil[i, 5] + q_airfoil[i+1, 5])
        Fx += p_mid*airfoil_cells[i].n1[0]*airfoil_cells[i].Ds1
        Fy += p_mid*airfoil_cells[i].n1[1]*airfoil_cells[i].Ds1
        
        x_mid = 0.5*(airfoil_cells[i].x1 + airfoil_cells[i].x2)
        y_mid = 0.5*(airfoil_cells[i].y1 + airfoil_cells[i].y2)
        M += p_mid*(-(x_mid-x_ref)*airfoil_cells[i].n1[1] + (y_mid-y_ref)*airfoil_cells[i].n1[0])*airfoil_cells[i].Ds1
    
    L = Fy*np.cos(alpha) - Fx*np.sin(alpha)
    D = Fy*np.sin(alpha) + Fx*np.cos(alpha)
    
    C_L = L/(0.5*rho_inf*U_inf**2*chord)
    C_D = D/(0.5*rho_inf*U_inf**2*chord)
    C_M = M/(0.5*rho_inf*U_inf**2*chord**2)
    
    
    return cp_airfoil, C_L, C_D, C_M

def read_residual_history(file_name):
    data = pd.read_csv(file_name, sep=",")
    # rename the columns for easier access
    data.columns = ['Time', 'Residual_0', 'Residual_1', 'Residual_2', 'Residual_3', 'cl', 'cd', 'cm']
    Time = data['Time'].values
    R0 = data['Residual_0'].values
    R1 = data['Residual_1'].values
    R2 = data['Residual_2'].values
    R3 = data['Residual_3'].values
    cl = data['cl'].values
    cd = data['cd'].values
    cm = data['cm'].values
    return Time, R0, R1, R2, R3, cl, cd, cm
