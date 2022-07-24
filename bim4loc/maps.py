import numpy as np
from TemporalParticleFilter.geometry import pose2
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.patches import Polygon

class Map:
        #http://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html
        def __init__(self,patches2d,rayCastingScene):
            self.scene =  rayCastingScene #o3d.cuda.pybind.t.geometry.RaycastingScene, for actual measurement model
            self.patches2d = patches2d #{vertices,faces,color}, for drawing only

        def forward_measurement_model(self, x : pose2, angles : np.ndarray):
            zmax = 10.0

            # http://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html
            rays = o3d.core.Tensor([[x.x,x.y,0,np.cos(x.theta+a),np.sin(x.theta+a),0] for a in angles],
                       dtype=o3d.core.Dtype.Float32)
            ans = self.scene.cast_rays(rays)
            z = ans['t_hit'].numpy()
            z[z == np.inf] = zmax
            return z.reshape(-1,1)

        def show(self,ax : plt.Axes = None, xrange = None, yrange = None):
            if ax == None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_xlabel('x'); ax.set_ylabel('y'); 
                ax.set_aspect('equal'); ax.grid()

            for patch in self.patches2d:
                triangles = patch["triangles"]
                for tri in triangles:
                    polygon = Polygon(tri, True)
                    ax.add_patch(polygon)

            if xrange is not None: ax.set_xlim(xrange)
            if yrange is not None: ax.set_ylim(yrange)

            return ax




