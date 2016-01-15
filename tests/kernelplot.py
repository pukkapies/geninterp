import sys, os
current_path = os.getcwd()
current_path_list = current_path.split('/')
module_path_list = current_path_list[:-1]
module_path = '/'.join(module_path_list)

sys.path.append(module_path)

from geninterp import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')

K = Wendland(6,4,c=0.3,dimension=2)

#K = Gaussian(0.5, dimension=2)

xrange = np.arange(-2, 2, 0.1)
yrange = np.arange(-2, 2, 0.1)
xgrid, ygrid = np.meshgrid(xrange, yrange)

centre = np.array([0,0])

#W = 5765760 * K.eval(np.array([xgrid.T, ygrid.T]).T, centre)
W =  K.eval(np.array([xgrid.T, ygrid.T]).T, centre)

surf = ax.plot_surface(xgrid, ygrid, W, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
#ax.set_zlim3d(-0.5, 6.01)
fig.colorbar(surf, shrink=0.5, aspect=10)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(xgrid, ygrid, W, rstride=2, cstride=2)

plt.show()
