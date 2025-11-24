
import HSPM,geometryGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creating the geometry and source panels
panels = geometryGenerator.GenerateNACA4digit(maxCamber=2.0,
                                              positionOfMaxCamber=4.0,
                                              thickness=12.0,
                                              pointsPerSurface=120)

# Instantiating HSPM class to compute the pressure solution on the given geometry
prob = HSPM.HSPM(listOfPanels = panels, alphaRange = np.arange(-5.0,16.0,0.5))
# Solving...
prob.run()


print(prob.deltaCPvalarezo)

(alphaMax, clMax)=prob.findAlphaMaxClMax(valarezoCriterion=5.0)
print("alphaMax: %.3lf, clMax: %.3lf" % (alphaMax, clMax))

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

df.to_csv("polars/naca2412_polars_results.csv", index=False)
