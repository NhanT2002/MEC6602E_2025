from liftPolar import ClPolar, CdPolar, getBuffetCiterion
import numpy as np

fOut = open("liftPolar.dat", "w")

for alpha in np.arange(-5, 10, 0.1):
    cl = ClPolar(alpha)
    cd = CdPolar(alpha)
    buffet = getBuffetCiterion(alpha)
    fOut.write("%f %f %lf %f\n" %(alpha,cl, cd, buffet))

fOut.close()
