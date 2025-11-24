
import HSPM,geometryGenerator

# Creating the geometry and source panels
panels = geometryGenerator.GenerateNACA4digit(maxCamber=0.0,
                                              positionOfMaxCamber=0.0,
                                              thickness=12.0,
                                              pointsPerSurface=50)

# Instantiating HSPM class to compute the pressure solution on the given geometry
prob = HSPM.HSPM(listOfPanels = panels, alphaRange = [2.5])
# Solving...
prob.run()

# Extracting tangential velocity on lower surface for boundary layer integration
(lowerCoords, lowerV) = prob.getLowerVtangential()

print(lowerCoords[0])
print(lowerV)

# Extracting tangential velocity on upper surface for boundary layer integration
(upperCoords, upperV) = prob.getUpperVtangential()

print(upperCoords[0])
print(upperV)