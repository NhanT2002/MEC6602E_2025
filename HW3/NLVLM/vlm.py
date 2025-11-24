import numpy as np
from Vector3 import Vector3
from vortexRing import vortexRing as panel
from math import *
from scipy import special
import sys
from scipy import linalg
from Polar import ClPolar, CdPolar, getTestCriterion
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d
import os

class VLM:
    def __init__(self, ni=5, nj=10, chordEta=[0.0, 1.0], chordVal=[1.0, 1.0], twistEta=[0.0,1.0], twist=[0.0,0.0], span=5.0, sweep=30.0, Sref = 1.0, referencePoint=[0.25,0.0,0.0], wingType="rect", distr="cosine", alphaRange = [3.0], clTarget = 0.0, Sym=True, Periodic=False, period=0.0, nperiod=0, mu=0.0, mu4 = 0.0, omega=1.0, maxIter=1000, nonLinear=False, prefix = "", tol=1.e-6, minIter=5, viscosity=0.0, density = 1.0, speed = 1.0, isoCl = False):

        self.nonLinear = nonLinear
        self.prefix = prefix


        self.size = ni * nj

        self.A   = np.zeros((self.size,self.size))
        self.Em1   = np.zeros((self.size,nj))
        self.E    = np.zeros((self.size,nj))
        self.rhs = np.zeros(self.size)
        self.inducedDownwash = [np.zeros((self.size,self.size)), np.zeros((self.size,self.size)), np.zeros((self.size,self.size))]

        self.nw = 1
        self.panels     = []
        self.wakePanels = []
        self.ni = ni
        self.nj = nj
        self.mu = mu
        self.mu4 = mu4
        self.omega = omega
        self.maxIter=maxIter
        self.minIter=minIter
        self.tol=tol

        self.gamma   = np.zeros(self.size)
        self.gammaij = np.zeros(self.size)
        self.alphaCor = np.zeros(self.nj)
        self.dAlpha   = np.zeros(self.nj)
        self.cl_sec   = np.zeros(self.nj)
        self.localChord = np.ones(self.nj+1)
        self.centerChord = np.ones(self.nj)

        self.liftAxis = Vector3(0.0,0.0,1.0)

        self.Sref      = Sref
        self.chordEta = chordEta
        self.chordVal = chordVal
        self.chordRoot = self.chordVal[0]
        self.chordTip  = self.chordVal[1]
        #self.cavg      = 0.5 * (chordRoot + chordTip)
        self.twistEta = twistEta
        self.twist  = twist

        self.isoCl = isoCl

        self.span      = span
        self.sweep     = sweep * pi / 180.0
        self.referencePoint = Vector3(referencePoint[0],
                                      referencePoint[1],
                                      referencePoint[2])
        self.wingType = wingType
        self.distr    = distr
        self.sym    = Sym
        self.periodic    = Periodic
        self.period = period
        self.nperiod = nperiod

        self.AOA    = []
        self.CL     = []
        self.CDw     = []
        self.CDp    = []
        self.CDi_LL = []
        self.Oswald = []
        self.test = []
        self.CM     = []
        self.spanLoad = []

        self.alphaRange = alphaRange
        self.clTarget = clTarget
        self.Ufree = Vector3(1.0,0.0,0.0)
        self.rho = 1.0

        # Dimensiona values
        self.density_dim = density
        self.viscosity_dim = viscosity
        self.speed_dim = speed

    def calcA(self):

        self.A *= 0.0
        self.inducedDownwash[0] *= 0.0
        self.inducedDownwash[1] *= 0.0
        self.inducedDownwash[2] *= 0.0

        for j in range(0,self.nj):
            for i in range(0,self.ni):

                ia = self.ni*j + i
                #print("Effect of panel %d" %(ia))
                panel = self.panels[ia]
                collocationPoint = panel.center
                normal = panel.normal
                #print(collocationPoint)
                for j2 in range(0,self.nj):
                    for i2 in range(0,self.ni):
                        ia2 = self.ni*j2+i2

                        panel2 = self.panels[ia2]

                        u = panel2.influence(collocationPoint, Sym=self.sym)
                        downWash = panel2.influence(collocationPoint, Sym=self.sym, boundInfluence=False)


                        # Ajout de l'influence du sillage
                        if (i2 == self.ni-1):
                            for n in range(0,self.nw):
                                iaw = self.nw*j2 + n
                                wakePanel = self.wakePanels[iaw]

                                u += wakePanel.influence(collocationPoint, Sym=self.sym)
                                downWash += wakePanel.influence(collocationPoint, Sym=self.sym, boundInfluence=False)

                        self.A[ia,ia2] += u.dot(normal)

                        self.inducedDownwash[0][ia,ia2] += downWash[0]
                        self.inducedDownwash[1][ia,ia2] += downWash[1]
                        self.inducedDownwash[2][ia,ia2] += downWash[2]


    def calcRHS(self):
        for i,r in enumerate(self.rhs):
            self.rhs[i] = -self.Ufree.dot(self.panels[i].normal)
            # self.rhs[i] = 0.0
            # for indexPeriodic in range(-self.nperiod,self.nperiod+1):
            #     self.rhs[i] -= self.Ufree.dot(self.panels[i].normal)

    def solve(self):
        self.gamma = np.linalg.solve(self.A,self.rhs)


    def postProcess(self):
        for j in range(0,self.nj):
            for i in range(0,self.ni):
                ia = self.ni*j + i
                if (i == 0):
                    self.gammaij[ia] = self.gamma[ia]
                else:
                    iam = ia - 1
                    self.gammaij[ia] = self.gamma[ia] - self.gamma[iam]

    def computeForcesAndMoment(self):
        self.CL.append(0.0)
        self.CM.append(0.0)
        self.CDw.append(0.0)
        self.CDp.append(0.0)
        self.test.append(0.0)
        inducedDownwashX = np.dot(self.inducedDownwash[0],self.gamma)
        inducedDownwashY = np.dot(self.inducedDownwash[1],self.gamma)
        inducedDownwashZ = np.dot(self.inducedDownwash[2],self.gamma)

        for index,panel in enumerate(self.panels):
            force = self.Ufree.crossProduct(panel.dl()) * self.rho * self.gammaij[index]

            distToRefrence = self.referencePoint - panel.forceActingPoint()
            moment = force.crossProduct(distToRefrence)

            downWash = Vector3(inducedDownwashX[index], inducedDownwashY[index], inducedDownwashZ[index])

            self.CL[-1] += force.dot(self.liftAxis)
            self.CM[-1] += moment[1]
            #self.CDw[-1] -= self.rho * downWash.dot(self.liftAxis) * self.gammaij[index] * panel.dy()

        self.CL[-1] /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * self.Sref)# * float(self.nperiod*2+1))
        #self.CDw[-1] /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * self.Sref)# * float(self.nperiod*2+1))
        self.CM[-1] /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * self.Sref)# * self.cavg * float(self.nperiod*2+1))

        for j in range(self.nj):
            area = 0.0
            cl = 0.0
            for i in range(self.ni):
                ia = self.ni*j + i
                panel = self.panels[ia]
                area += panel.Area()


            self.CDw[-1] += CdPolar((self.cl_sec[j]/(2.*pi) - self.alphaCor[j])*180.0/pi) *area
            Rec = (self.density_dim*self.centerChord[j]*self.speed_dim)/self.viscosity_dim
            #print(Rec)
            # Factor 2 for the upper and lower side
            self.CDp[-1] += 2.0*1.4*0.455/(log10(Rec)**2.58)*area
            self.test[-1] += getTestCriterion((self.cl_sec[j]/(2.*pi) - self.alphaCor[j])*180.0/pi)

        self.CDw[-1] /= self.Sref
        self.CDp[-1] /= self.Sref

    def computeInducedDragLiftingLine(self, verbose=False):
        self.CDi_LL.append(0.0)
        self.Oswald.append(0.0)
        (y, cl_loc, gamma_sec, area_loc, dy) = self.getSpanload()
        #print(y)
        #print(np.flip(-y))
        y2 = np.concatenate((np.flip(-y), y))

        theta = np.arccos(-1.0*y/self.span)

        size_system = self.nj

        A = np.zeros((size_system, size_system))
        X = np.zeros(size_system)
        B = np.zeros(size_system)

        #B = np.concatenate((np.flip(gamma_sec), gamma_sec))
        B = gamma_sec

        for i in range(size_system):
            for j in range(size_system):
               n=2*j+1
               A[i,j]=4.0*self.Ufree.Magnitude()*self.span*np.sin(n*theta[i])



        #Solving for the An  coefficients
        X = np.linalg.solve(A, B)


        delta = 0.0
        for i in range(1,size_system):
           n=2*i+1
           delta+=n*(  (X[i]/X[0])**2  )

        #print(delta)

        self.Oswald[-1] =    1.0/(1.0+delta)

        AR_tot = (2.0 * self.span)**2 / (2.0 * self.Sref)
        #print(self.CL[-1])
        #print(AR_tot)
        #print(e)
        self.CDi_LL[-1]=self.CL[-1]**2 / (np.pi * AR_tot * self.Oswald[-1])


    def writeSpanload(self,outputfile):
        print(outputfile)
        ypos = np.zeros(self.nj)

        for j in range(self.nj):
            area = 0.0
            cl = 0.0
            for i in range(self.ni):
                ia = self.ni*j + i
                panel = self.panels[ia]
                area += panel.Area()
                force = self.Ufree.crossProduct(panel.dl()) * self.rho * self.gammaij[ia]
                cl += force.dot(self.liftAxis)

            #cl /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * float(self.nperiod*2+1)*area)
            cl /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * area)

            ypos[j] = self.panels[self.ni * j].forceActingPoint()[1]
            self.cl_sec[j] = cl


        fid = open(outputfile, 'w')
        fid.write("VARIABLES= \"Y\",\"eta\",\"Cl\",\"Cl c\",\"Cl/c\",\"c\",\"d Alpha\",\"alphaE\", \"Cl(alphaE)\" \"Cd(alphaE)\"\n")
        for i,y in enumerate(ypos):
        #    fid.write("%.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf\n" % (y, self.cl_sec[i], self.cl_sec[i]*0.5*(self.localChord[i]+self.localChord[i+1]), self.cl_sec[i]/(0.5*(self.localChord[i]+self.localChord[i+1])), (self.localChord[i]+self.localChord[i+1]), self.alphaCor[i], self.cl_sec[i]/(2.*pi) - self.alphaCor[i]))
            AoAE = (self.cl_sec[i]/(2.*pi) - self.alphaCor[i])*180.0/pi
            chord = self.centerChord[i]
            fid.write("%.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %0.12lf %0.12lf\n" % (y, y/self.span, self.cl_sec[i], self.cl_sec[i]*chord, self.cl_sec[i]/chord, chord, self.alphaCor[i]*180.0/pi, AoAE, ClPolar(AoAE), CdPolar(AoAE)))
        fid.close()

    def getSpanload(self):
        ypos = np.zeros(self.nj)
        dy = np.zeros(self.nj)

        gamma_sec = np.zeros(self.nj)
        area_sec =  np.zeros(self.nj)
        for j in range(self.nj):
            area = 0.0
            cl = 0.0

            for i in range(self.ni):
                ia = self.ni*j + i
                panel = self.panels[ia]
                area += panel.Area()
                force = self.Ufree.crossProduct(panel.dl()) * self.rho * self.gammaij[ia]


                cl += force.dot(self.liftAxis)

            #cl /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * float(self.nperiod*2+1) * area)
            cl /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * area)
            area_sec[j]=area

            ypos[j]     = self.panels[self.ni * j].forceActingPoint()[1]
            dy[j]          = self.panels[self.ni * j].dy()
            self.cl_sec[j]     = cl


        self.ypos = ypos

        gamma_sec = 0.5*self.cl_sec*area_sec/dy
        return(ypos, self.cl_sec, gamma_sec, area_sec, dy)

    def writeSolution(self,outputfile):
        out = 'Variables=\"X\",\"Y\",\"Z\",\"GAMMA\"\n'
        out += 'ZONE T=\"WING\" i=%d,j=%d,k=1, ZONETYPE=Ordered\nDATAPACKING=BLOCK\nVARLOCATION=([4]=CELLCENTERED)\n'%(self.ni+1,self.nj+1)

        for j in range(0,self.nj):
            for i in range(0,self.ni):
                ia = self.ni*j + i
                pan = self.panels[ia]
                out += '%lf\n '%(pan.p1.x)
            out += '%lf\n '%(pan.p2.x)

        for i in range(0,self.ni):
            ia = self.ni*j + i
            pan = self.panels[ia]
            out += '%lf\n '%(pan.p4.x)
        out += '%lf\n'%(pan.p3.x)

        for j in range(0,self.nj):
            for i in range(0,self.ni):
                ia = self.ni*j + i
                pan = self.panels[ia]
                out += '%lf\n '%(pan.p1.y)
            out += '%lf\n '%(pan.p2.y)

        for i in range(0,self.ni):
            ia = self.ni*j + i
            pan = self.panels[ia]
            out += '%lf\n '%(pan.p4.y)
        out += '%lf\n'%(pan.p3.y)

        for j in range(0,self.nj):
            for i in range(0,self.ni):
                ia = self.ni*j + i
                pan = self.panels[ia]
                out += '%lf\n '%(pan.p1.z)
            out += '%lf\n '%(pan.p2.z)

        for i in range(0,self.ni):
            ia = self.ni*j + i
            pan = self.panels[ia]
            out += '%lf\n '%(pan.p4.z)
        out += '%lf\n'%(pan.p3.z)

        for j in range(0,self.nj):
            for i in range(0,self.ni):
                ia = self.ni*j + i
                pan = self.panels[ia]
                out += '%lf\n '%(self.gamma[ia])


        f = open(outputfile,'w')
        f.write(out)
        f.close()

    def writeData(self, iter, outputfile):

        ypos = np.zeros(self.nj)

        for j in range(self.nj):
            area = 0.0
            cl = 0.0
            for i in range(self.ni):
                ia = self.ni*j + i
                panel = self.panels[ia]
                area += panel.Area()
                force = self.Ufree.crossProduct(panel.dl()) * self.rho * self.gammaij[ia]
                cl += force.dot(self.liftAxis)

            #cl /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * float(self.nperiod*2+1)*area)
            cl /= ( 0.5 * self.rho * self.Ufree.Magnitude()**2 * area)

            ypos[j] = self.panels[self.ni * j].forceActingPoint()[1]
            self.cl_sec[j] = cl

        if iter == 1:
            fid = open(outputfile, 'w')
            fid.write("VARIABLES= \"Iter\",\"Y\",\"Cl\",\"dAlpha\"\n")
        else:
            fid = open(outputfile, 'a')

        fid.write("ZONE\n")
        #fid.write("ZONE=dede\n")
        for i,y in enumerate(ypos):
            fid.write("%d %12.6le %12.6le %12.6le\n" % (iter, y, self.cl_sec[i], self.alphaCor[i]))

        fid.close()

    def initializeWing(self):
        dy = self.span/float(self.nj)
        if self.sym == True:
            y = 0.0
        else:
            y = -self.span/2.0

        y = 0.0
        yNext = y + dy

        if len(self.twistEta)==2:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'linear')
            chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'linear')
        elif len(self.twistEta)==3:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'quadratic')
            chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'quadratic')
        else:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'cubic')
            chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'cubic')

        for j in range(self.nj):
            eta     = y / self.span
            etaNext = yNext / self.span

            twist = twistSpline(eta)#(1.0 - eta) * self.twistRoot + eta * self.twistTip
            twistNext = twistSpline(etaNext)#(1.0 - etaNext) * self.twistRoot + etaNext * self.twistTip

            twist *= -1.0
            twistNext *= -1.0

            chord = Vector3(chordSpline(eta), 0.0, 0.0).rotate(0.0,twist,0.0)
            chordNext = Vector3(chordSpline(etaNext), 0.0, 0.0).rotate(0.0,twist,0.0)
            #chord = Vector3((1.0 - eta) * self.chordRoot + eta * self.chordTip, 0.0, 0.0).rotate(0.0,twist,0.0)
            #chordNext = Vector3((1.0 - etaNext) * self.chordRoot + etaNext * self.chordTip, 0.0, 0.0).rotate(0.0,twistNext, 0.0)

            #print(chord)

            pt = Vector3(tan(self.sweep) * y, y, 0.0)
            ptNext = Vector3(tan(self.sweep) * yNext, yNext, 0.0)

            ds = chord / float(self.ni)
            dsNext = chordNext / float(self.ni)
            #print(ds)

            self.localChord[j] = chord.x
            self.localChord[j+1] = chordNext.x
            for i in range(self.ni):
                p1 = copy.deepcopy(pt)
                p4 = copy.deepcopy(ptNext)

                pt = pt + ds
                ptNext += dsNext

                p2 = copy.deepcopy(pt)
                p3 = copy.deepcopy(ptNext)
                #print(p1)
                #print(p2)
                #print(p3)
                #print(p4)
                self.panels.append(panel(p1,p2,p3,p4))



            y += dy
            yNext += dy





    def initializeWingElliptic(self):

        if self.distr=="cosine":
            #print(self.distr)
            if self.sym:
               theta_i = np.linspace(np.pi/2., np.pi, self.nj+1)
            else:
               theta_i = np.linspace(0, np.pi, self.nj+1)
            y_i = -self.span*np.cos(theta_i)
        elif self.distr=="regular":
            y_i = np.linspace(0,self.span,self.nj+1)

        if len(self.twistEta)==2:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'linear')
            #chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'linear')
        elif len(self.twistEta)==3:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'quadratic')
            #chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'quadratic')
        else:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'cubic')
            #chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'cubic')
        for j in range(self.nj):
            y     = y_i[j]
            yNext = y_i[j+1]
            eta     = y / self.span
            etaNext = yNext / self.span

            twist = twistSpline(eta)#(1.0 - eta) * self.twistRoot + eta * self.twistTip
            twistNext = twistSpline(etaNext)#(1.0 - etaNext) * self.twistRoot + etaNext * self.twistTip

            twist *= -1.0
            twistNext *= -1.0

            chord = Vector3(np.sqrt(1.0 - eta*eta*0.99) * self.chordRoot, 0.0, 0.0).rotate(0.0,twist,0.0)

            self.localChord[j] = chord.x
            self.localChord[j+1] = chord.x
            chordNext = Vector3(np.sqrt(1.0 - etaNext*etaNext*0.99) * self.chordRoot, 0.0, 0.0).rotate(0.0,twistNext,0.0)
            #chordNext = Vector3(0.0, 0.0, 0.0).rotate(0.0,twistNext,0.0)
            self.localChord[j+1] = chordNext.x

            pt = Vector3(tan(self.sweep) * y, y, 0.0) + (Vector3(self.chordRoot,0.0,0.0)-chord)*0.5
            ptNext = Vector3(tan(self.sweep) * yNext, yNext, 0.0) + (Vector3(self.chordRoot,0.0,0.0)-chordNext)*0.5

            #if self.distr=="cosine":

            ds = chord / float(self.ni)
            dsNext = chordNext / float(self.ni)

            for i in range(self.ni):
                p1 = pt
                p4 = ptNext

                pt = pt + ds
                ptNext += dsNext

                p2 = pt
                p3 = ptNext

                self.panels.append(panel(p1,p2,p3,p4))

    def initializeWingCosine(self):

        theta_i = np.linspace(0.5*np.pi, np.pi, self.nj+1)
        y_i = -self.span*np.cos(theta_i)
        dy = self.span/float(self.nj)

        if len(self.twistEta)==2:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'linear')
        elif len(self.twistEta)==3:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'quadratic')
        else:
            twistSpline = interp1d(self.twistEta, self.twist, fill_value = "extrapolate", kind = 'cubic')


        if len(self.chordEta)==2:
            chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'linear')
        elif len(self.chordEta)==3:
            chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'quadratic')
        else:
            chordSpline = interp1d(self.chordEta, self.chordVal, fill_value = "extrapolate", kind = 'cubic')

        for j in range(self.nj):
            y     = y_i[j]
            yNext = y_i[j+1]
            eta     = y / self.span
            etaNext = yNext / self.span

            twist = twistSpline(eta)#(1.0 - eta) * self.twistRoot + eta * self.twistTip
            twistNext = twistSpline(etaNext)#(1.0 - etaNext) * self.twistRoot + etaNext * self.twistTip

            twist *= -1.0
            twistNext *= -1.0

            chord = Vector3(chordSpline(eta), 0.0, 0.0).rotate(0.0,twist,0.0)
            chordNext = Vector3(chordSpline(etaNext), 0.0, 0.0).rotate(0.0,twist,0.0)
            #chord = Vector3((1.0 - eta) * self.chordRoot + eta * self.chordTip, 0.0, 0.0).rotate(0.0,twist,0.0)
            #chordNext = Vector3((1.0 - etaNext) * self.chordRoot + etaNext * self.chordTip, 0.0, 0.0).rotate(0.0,twistNext,0.0)

            # Twist around the quarterChord
            quarterChord =Vector3(0.0, 0.25*chord.y, 0.25*chord.z)
            quarterChordNext = Vector3(0.0, 0.25*chordNext.y, 0.25*chordNext.z)



            pt = Vector3(tan(self.sweep)* y, y, 0.0)
            ptNext = Vector3(tan(self.sweep)* yNext, yNext, 0.0)

            pt-=quarterChord
            ptNext-=quarterChordNext

            ds = chord / float(self.ni)
            dsNext = chordNext / float(self.ni)

            self.localChord[j] = chord.x
            self.localChord[j+1] = chordNext.x


            for i in range(self.ni):
                p1 = copy.deepcopy(pt)
                p4 = copy.deepcopy(ptNext)

                pt = pt + ds
                ptNext += dsNext

                p2 = pt
                p3 = ptNext

                self.panels.append(panel(p1,p2,p3,p4))


    def initializeWake(self):
        i = self.ni-1
        for j in range(0,self.nj):
            ia = self.ni*j + i
            pan = self.panels[ia]
            p1 = pan.p2
            p4 = pan.p3
            p2 = p1 + self.Ufree * 100.0 * self.chordRoot
            p3 = p4 + self.Ufree * 100.0 * self.chordRoot
            self.wakePanels.append(panel(p1,p2,p3,p4))

    def updateWake(self):
        i = self.ni-1
        for j in range(0,self.nj):
            ia = self.ni*j + i
            pan = self.panels[ia]
            p1 = pan.p2
            p4 = pan.p3
            p2 = p1 + self.Ufree * 100.0 * self.chordRoot
            p3 = p4 + self.Ufree * 100.0 * self.chordRoot
            self.wakePanels[j] = panel(p1,p2,p3,p4)


    def updateFreeStream(self,alpha):
        self.Ufree = Vector3(cos(alpha * pi / 180.0), 0.0, sin(alpha * pi / 180.0))
        self.liftAxis = Vector3(-sin(alpha * pi / 180.0), 0.0, cos(alpha * pi / 180.0))


    def nonLinearCoupling(self):

        rms = 0.0

        for j in range(0,self.nj):

            clInviscid = self.cl_sec[j]
            #print(clInviscid)
            alphaEffective = clInviscid/(2.*pi) - self.alphaCor[j]# + alpha
            #print (self.panels[self.ni * j].forceActingPoint()[1])
            clVisc = ClPolar(alphaEffective*180.0/pi)
            self.dAlpha[j] = (clVisc-clInviscid)/(2.*pi)


            if self.sym == True:
               self.dAlpha[0] += self.mu*(self.alphaCor[0]-2.0*self.alphaCor[0]+self.alphaCor[1])

            for j in range(1,self.nj-1):
                self.dAlpha[j] += self.mu*(self.alphaCor[j-1]-2.0*self.alphaCor[j]+self.alphaCor[j+1])

        for j in range(0,self.nj):
            self.dAlpha[j] *= self.omega
            self.alphaCor[j] = self.alphaCor[j] + self.dAlpha[j]
            rms += self.dAlpha[j]**2
            for i in range(0,self.ni):
                ia = self.ni*j + i
                normal = self.panels[ia].normal.rotate(0.0,self.dAlpha[j]*180./pi,0.0)
                self.panels[ia].normal = normal

        rms = sqrt(rms)/float(self.nj)
        return rms
        #print(self.alphaCor)


    def run(self, graph = False, writeHistory = False, saveEvery=1):

        if graph:
            fig = plt.figure()
            plt.ion()

        if self.wingType == "rect":
            self.initializeWing()
        elif self.wingType == "cosine":
             self.initializeWingCosine()
        elif self.wingType == "fullEllipse":
            self.initializeWingElliptic()

        else:
            print("Wrong Input, defaulting to regular wing discretization")
            self.initializeWing()

        for j in range(0,self.nj):
            for i in range(0,self.ni):
                angle = 1.0e-6*sin(pi/self.nj * j)

                ia = self.ni*j + i
                self.panels[ia].normal = self.panels[ia].normal.rotate(0.0,angle,0.0)
        for j in range(0,self.nj-1):
            self.centerChord[j] = 0.5*(self.localChord[j]+self.localChord[j+1])

        #print(self.panels[ia].normal)

        self.initializeWake()

        with open("force_%s.dat"%(self.prefix), "w") as fOut:
            fOut.write("VARIABLES = AoA, CL, CM, CDw, CDp, CDi_ll, CDtot, Lift Dw, Dp, Di, Dtot, Oswald, Buffet/Stall\n")

        #Initialze Drag Arrray
        Dw = []
        Di = []
        Dp = []
        Dtot = []
        Lift = []

        for alpha in self.alphaRange:
            ClIteration = 0

            while ClIteration == 0 or abs(self.CL[-1]-self.clTarget)>1.0e-3:
                if ClIteration == 1:
                    alpha+=0.1
                elif ClIteration >1:
                    print("CL error: %0.6lf" %(self.CL[-1]-self.clTarget))
                    alpha -= (self.CL[-1]-self.clTarget)/((self.CL[-1]-self.CL[-2])/(self.AOA[-1]-self.AOA[-2]))


                self.AOA.append(alpha)
                ClIteration+=1

                convFile = open("Conv_%s_a%04.2f.dat" %(self.prefix, alpha), "w")
                convFile.write("iteration, rms")
                self.updateFreeStream(alpha)
                self.updateWake()
                self.calcA()

                iter = 0
                rms = 1.

                while (rms> self.tol and iter<self.maxIter) or iter<self.minIter:
                    iter +=1
                    self.calcRHS()
                    self.solve()

                    self.postProcess()
                    self.getSpanload()
                    if self.nonLinear:
                        rms = self.nonLinearCoupling()
                    else:
                        rms = 0.0
                    if rms != rms:
                        print("Viscous coupling failed")
                        sys.exit()

                    print("%d   %le" %(iter, rms))
                    convFile.write("%d   %le\n" %(iter, rms))
                    convFile.flush()

                    if graph:
                        fig.clear()
                        plt.xlabel('y')
                        plt.ylabel('Cl c')
                        plt.plot(self.ypos,self.cl_sec)

                        plt.show()
                        plt.pause(0.00000001)

                    if writeHistory and iter % saveEvery == 0:
                        self.writeData(iter, "History_%s_Cl%04.2f.dat" %(self.prefix, self.CL[-1]))



                self.postProcess()
                self.computeForcesAndMoment()
                self.computeInducedDragLiftingLine(verbose=True)

                Dw.append(0.5*self.Sref*2*self.density_dim*self.speed_dim**2*self.CDw[-1])
                Di.append(0.5*self.Sref*2*self.density_dim*self.speed_dim**2*self.CDi_LL[-1])
                Dp.append(0.5*self.Sref*2*self.density_dim*self.speed_dim**2*self.CDp[-1])
                Dtot.append(Dw[-1]+Di[-1]+Dp[-1])
                Lift.append(0.5*self.Sref*2*self.density_dim*self.speed_dim**2*self.CL[-1])
                print('Alpha= %.2lf CL= %.3lf CM= %.4lf CDw= %.4lf CDp = %0.4lf CDi_ll= %.4lf CDtot= %.4lf Oswald = %lf Buffet = %f' % (alpha, self.CL[-1], self.CM[-1], self.CDw[-1]*10000.0, self.CDp[-1]*10000.0, self.CDi_LL[-1]*10000.0, (self.CDw[-1]+self.CDp[-1]+self.CDi_LL[-1])*10000.0 , self.Oswald[-1], self.test[-1]))
                print('Alpha= %.2lf CL= %.3lf CM= %.4lf Dw= %.4lf Dp = %0.4lf Di_ll= %.4lf Dtot= %.4lf Oswald = %lf Buffet = %f' % (alpha, self.CL[-1], self.CM[-1], Dw[-1], Dp[-1], Di[-1], Dtot[-1] , self.Oswald[-1], self.test[-1]))

                if self.isoCl == False:
                    self.clTarget = self.CL[-1]

            self.writeSpanload("Spanload_%s_Cl%04.2f_AoA%04.2f.dat" %(self.prefix, self.CL[-1], alpha))
            self.writeSolution("3D_sol_%s_Cl%04.2f_AoA%04.2f.dat" %(self.prefix, self.CL[-1],alpha))

            convFile.close()


            with open("force_%s.dat"%(self.prefix), "a") as fOut:
                fOut.write('%8.6lf %8.6lf  %8.6lf %8.6lf %8.6lf %8.6lf %8.6lf %8.6lf %lf %lf %lf %lf %lf %lf\n' % (alpha, self.CL[-1],self.CM[-1], self.CDw[-1]*10000.0, self.CDp[-1]*10000.0, self.CDi_LL[-1]*10000.0, (self.CDw[-1]+self.CDp[-1]+self.CDi_LL[-1])*10000.0, Lift[-1], Dw[-1], Dp[-1], Di[-1], Dtot[-1], self.Oswald[-1], self.test[-1]))

        return Lift, Dtot, self.test[-1]


if __name__ == '__main__':
    prob = VLM(ni=5,
               nj=20,
               twistEta=0.0,
               twist=0.0,
               span=5.0,
               sweep=0.0,
               Sref =3.250,
               referencePoint=[0.25,0.0,0.0],
               wingType=1,
               alphaRange = [0.0,5.0, 10.0, 20.0])
    prob.run()
