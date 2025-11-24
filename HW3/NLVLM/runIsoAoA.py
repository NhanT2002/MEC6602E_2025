
# -*- coding: utf-8 -*-
import numpy as np
from vlm  import *
from math import pi


# Dimensional lift
lift = 45000*9.81

# Air property at 38 000 ft
density = 0.331985 # kg/m³
soundSpeed = 295.070  # m/s
viscosity = 0.0000143226 # Pa.s

Mach = 0.78
speed = Mach*soundSpeed # m/s

drag = []
conf = []

chordRoot =3.0
minDrag = -999.999
minIndex = 0

for chordRoot in [3.5]:
    for chordMid in [2.0]:
        for midTwist in [2.0]:
            for tipTwist in [2.5]:

                span = 25.0
                Sref = 100.0


                Sinboard = (chordRoot+chordMid)*span*0.5
                chordTip = (Sref-Sinboard)/span/0.5-chordMid
                Sref = Sinboard+(chordMid+chordTip)*span/2.0

                Cltarget = lift/(0.5*density*speed**2*Sref)
                prefix = "Optim_chordTip%4.2f_chordRoot%4.2f_chordMid%4.2f_midTwist%4.2f_tipTwist%4.2f" %(chordTip, chordRoot, chordMid, midTwist, tipTwist)
                #Sref = (chordRoot+chordMid)*etaMid*span + (chordTip+chordMid)*(1-etaMid)*span

                if chordTip>0.1:


                    Ar = (span)**2/Sref
                    print("\n##########################################################")
                    print("Configuraton : %s" %prefix)
                    print("Cltarget : %lf" %Cltarget)
                    print("Root Chord : %lf" %chordRoot)
                    print("Mid Chord : %lf" %chordMid)
                    print("Tip Chord : %lf" %(chordTip))
                    print("Root Twist : %lf" %0.0)
                    print("Mid Twist : %lf" %midTwist)
                    print("Tip Twist : %lf" %(tipTwist))
                    print("Aspect Ratio : %lf" %Ar)

                    prob = VLM(ni=4, #number of mesh panel in I
                             nj=25,  #number of mesh panel in J
                             span=span/2.0,  # Run on a half wing with symmetry
                             sweep=25.0,     # Sweep angle
                             Sref = Sref/2.0, # Run on a half wing with symmetry
                             Sym = True,    # Activate the symmetry plane
                             nonLinear = True, # Activate the non linear VLM for coupling with the polar
                             wingType="cosine", # Function to have a trapezoid wing with cosine spacing towards the tip
                             alphaRange = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], # Value of AoA to fdo a 3D polar
                             mu = 0.01,    # Numerical dissipation in NL-VLM. Keep constant if no issue
                             twistEta = [0.0, 0.5, 1.0], # Twist distribution (interpolation is used between). Eta station are fraction of span
                             twist = [0.0, midTwist, tipTwist],
                             chordEta = [0.0, 0.5, 1.0], # Chord distribution (interpolation is used between)
                             chordVal = [chordRoot, chordMid, chordTip],
                             omega = 0.5,   # Under relaxation in non linear coupling
                             maxIter = 100, # Maximum number of non linear iteration
                             tol = 1.0e-5,  # Tolerance of the non linear coupling
                             prefix = prefix, # Prefix for the solution files
                             viscosity = viscosity, # Dimensional viscosity for profile drag model
                             density = density, # Dimensional density
                             speed = speed) # Dimensional speed

                    Cl, D, validity = prob.run(graph = False)

                    # Search for the minimum drag in proposed configuration
                    if validity <1.0e-6: # Test if the validity criterion was activated
                        conf.append(prefix)
                        drag.append(D[-1])

                        minDrag = min(drag)
                        minIndex= drag.index(minDrag)

                        print("Minimum drag = %lf" %minDrag)
                        print("Configuration : %s" %conf[minIndex])

if minDrag>-998:
    print("Minimum drag = %lf" %minDrag)
    print("Configuration : %s" %conf[minIndex])
else:
    print("No valid configuration found for current parametrization")
prob = None
