import HSPM,geometryGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creating the geometry and source panels
panels = geometryGenerator.ReadPoints("airfoil_points_sc20712.dat", 129)

# Instantiating HSPM class to compute the pressure solution on the given geometry
prob = HSPM.HSPM(listOfPanels = panels, alphaRange = np.arange(-5.0,16.0,0.5))
# Solving...
prob.run()
# Extracting alpha max and cl max with Valarezo
alphaMax, clMax = prob.findAlphaMaxClMax(valarezoCriterion=5.0)
print('AlphaMax= %.2lf ClMax= %.4lf' % (alphaMax,clMax))

valarezo = np.zeros_like(prob.alphaRange)
for i in range(len(prob.alphaRange)):
    alpha = prob.alphaRange[i]
    if alpha < alphaMax:
        valarezo[i] = 0
    else:
        valarezo[i] = 1
    

df = pd.DataFrame({
    'aoa': prob.alphaRange,
    'cl': prob.CL,
    'cd': prob.CD,
    'cm': prob.CM,
    'valarezo': valarezo
})

df.to_csv("polars/sc20712_polars_results.csv", index=False)