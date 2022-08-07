import numpy as np
from bim4loc.geometry import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import PcdSolid, ifc_converter, PcdSolid, ArrowSolid
from bim4loc.agents import Drone
from bim4loc.maps import Map
from bim4loc.filters import vanila
import time
import logging

logging.basicConfig(format = '%(levelname)s %(lineno)d %(message)s')
logger = logging.getLogger().setLevel(logging.WARNING)

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = Pose2z(3,3,0, 1.5))
world = Map(solids)

straight = Pose2z(0.5,0,0,0)
turn_left = Pose2z(0,0,np.pi/8,0)
turn_right = Pose2z(0,0,-np.pi/8,0)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

model = world
_, _, extent = model.bounds()

Nparticles = 100
inital_poses = []
for i in range(Nparticles):
    inital_poses.append(
        Pose2z(np.random.uniform(drone.pose.y + extent[0]/10, drone.pose.y - extent[0]/10),
                np.random.uniform(drone.pose.y + extent[1]/10, drone.pose.y - extent[1]/10),
                np.random.uniform(-np.pi, +np.pi),
                0)
                        )
inital_poses[0] = Pose2z(drone.pose.x,drone.pose.y,drone.pose.theta,0.0); #<---------------------------------------- CHEATING
arrows = []
for i in range(Nparticles):
    arrows.append(ArrowSolid(name = f'arrow_{i}', alpha = 1/Nparticles, pose = inital_poses[i]))

pf = vanila(drone, model , inital_poses)
Z_STD = 0.05
Z_COV = np.kron(np.eye(drone.lidar_angles.size),Z_STD**2)
U_COV = 0.01 * np.diag([0.1,0.1,np.radians(0.1),0.0])

visApp = VisApp()
for s in solids:
    visApp.add_solid(s)
visApp.show_axes(True)
visApp.reset_camera_to_default()
[visApp.add_solid(a) for a in arrows]
visApp.add_solid(drone.solid)
pcd_scan = PcdSolid()
visApp.add_solid(pcd_scan)

time.sleep(0.1)
for t,u in enumerate(actions):
    drone.move(u)
    z, z_p = drone.scan(world, Z_STD)

    pf.step(z, Z_COV, u, U_COV)
    if t % 3 == 0:
        pf.resample()
    
    for a,p in zip(arrows, pf.particles):
        a.update_geometry(p)
        visApp.update_solid(a)
    pcd_scan.update(z_p.T)
    visApp.update_solid(drone.solid)
    visApp.update_solid(pcd_scan)

    time.sleep(0.1)

print('finished')
