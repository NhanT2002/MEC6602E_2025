import Generator.PyTree as G
import Converter.PyTree as C
import Transform.PyTree as T
import Geom.PyTree as D
import Converter.Internal as Internal

mesh11 = G.cartr2((0,0,0), (0.01,0.01,0), (1.,1.,0.0), (1.5,1.5,0.))
mesh12 = G.cartr2((1.5,0,0), (0.01,0.01,0), (1.1,1.0,0.0), (100,1.5,0.))
mesh21 = G.cartr2((0,1.5,0), (0.01,0.01,0), (1.0,1.1,0.0), (1.5,100,0.))
mesh22 = G.cartr2((1.5,1.5,0), (0.01,0.01,0), (1.1,1.1,0.0), (100,100,0.))

mesh11_12 = T.join(mesh11, mesh12)
mesh21_22 = T.join(mesh21, mesh22)
mesh1 = T.join(mesh11_12, mesh21_22)
mesh2 = T.rotate(mesh1, (0,0,0), (0,0,1), 90)
mesh3 = T.rotate(mesh1, (0,0,0), (0,0,1), 180)
mesh4 = T.rotate(mesh1, (0,0,0), (0,0,1), 270)
mesh5 = T.join(mesh1, mesh2)
mesh6 = T.join(mesh3, mesh4)
mesh = T.join(mesh5, mesh6)
mesh = T.reorder(mesh, (-2,-1,3))
mesh = T.translate(mesh, (0.5,0.,0.))

Internal.printTree(mesh)
C.convertPyTree2File(mesh, 'cartesian_mesh.cgns')

naca = D.naca('0012', N=1001)
C.convertPyTree2File(naca, 'naca0012_geom.cgns')