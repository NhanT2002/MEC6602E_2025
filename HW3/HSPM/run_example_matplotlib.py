
import HSPM,geometryGenerator
import matplotlib.pyplot as plt

# Creating the geometry and source panels
panels = geometryGenerator.GenerateNACA4digit(maxCamber=2.0,
                                              positionOfMaxCamber=4.0,
                                              thickness=12.0,
                                              pointsPerSurface=50)

# Instantiating HSPM class to compute the pressure solution on the given geometry
prob = HSPM.HSPM(listOfPanels = panels, alphaRange = [0.0,5.0,10.0,20.0,25.0])
# Solving...
prob.run()

# Generate plot
plt.plot(prob.alphaRange, prob.CL)
plt.xlabel('alpha [degrees]')
plt.ylabel('CL [-]')
plt.title('CL-alpha')
plt.grid()
plt.show() # show plot
# plt.savefig('CL_alpha.png') #save plot to file