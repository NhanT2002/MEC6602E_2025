import Converter.PyTree as C
import Geom.IBM as D_IBM
import Geom.PyTree as D
import Generator.PyTree as G

a = D.naca("0012")
D_IBM._setSnear(a,0.01)
a = D_IBM.setDfar(a, 10)
D_IBM._setIBCType(a, "slip")
D_IBM._setFluidInside(a)
octree = G.octree([a], [0.01], dfar=10.0, balancing=2)
C.convertPyTree2File(octree, 'naca_unstructured.cgns')