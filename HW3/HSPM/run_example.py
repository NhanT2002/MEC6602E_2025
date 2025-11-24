
import HSPM,geometryGenerator

# Creating the geometry and source panels
panels = geometryGenerator.GenerateNACA4digit(maxCamber=0.0,
                                              positionOfMaxCamber=0.0,
                                              thickness=6.0,
                                              pointsPerSurface=200)

# Instantiating HSPM class to compute the pressure solution on the given geometry
prob = HSPM.HSPM(listOfPanels = panels, alphaRange = [0.0,5.0,10.0,20.0,25.0])
# Solving...
prob.run()
# Extracting alpha max and cl max with Valarezo
alphaMax, clMax = prob.findAlphaMaxClMax(valarezoCriterion=14.0)
print('AlphaMax= %.2le ClMax= %.4le' % (alphaMax,clMax))
