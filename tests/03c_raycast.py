import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, PcdSolid, LinesSolid
from bim4loc.agents import Drone
from bim4loc.sensors import Lidar1D
from bim4loc.maps import RayCastingMap
import time
import keyboard

full_solids = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = Pose2z(3,3,0,1.5))
# sensor = Lidar1D(angles = np.array([0.2]))
sensor = Lidar1D()
sensor.piercing = False
sensor.max_range = 1000.0
drone.mount_sensor(sensor)

# solids = [s for s in full_solids if s.name == "1UH7XjeubFPe8ud33kpdAD" or s.name == "22fuoCLrXEA9lnNzqjOo6F"] 
world = RayCastingMap(full_solids)

straight = Pose2z(0.5,0,0,0)
turn_left = Pose2z(0,0,np.pi/8,0)
turn_right = Pose2z(0,0,-np.pi/8,0)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

visApp = VisApp()
for s in world.solids.values():
    visApp.add_solid(s,"world")
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

# visApp.add_O3DVisualizer('o3d_window','o3d_scene')
# for s in world.solids.values():
#         visApp.add_solid(s,"o3d_scene")
# visApp.redraw('o3d_scene')

visApp.add_solid(drone.solid)
pcd_scan = PcdSolid()
visApp.add_solid(pcd_scan)
line_scan = LinesSolid()
visApp.add_solid(line_scan)

time.sleep(1)
for a in actions:
    drone.move(a)
    z, z_solid_names, p = drone.scan(world, project_scan = True)
    # for s in world.solids.values():
    #     if s.name in z_solid_names:
    #         s.material.base_color = (1,0,0,1)
    #     else:
    #         s.material.base_color = np.hstack((s.ifc_color,1))

    #show rays
    p = np.hstack((drone.pose.t, p))
    pcd_scan.update(p.T)
    line_ids = np.zeros((p.shape[1],2), dtype = int)
    line_ids[:,1] = np.arange(p.shape[1])
    line_scan.update(p.T, line_ids)
    
    visApp.update_solid(pcd_scan)
    visApp.update_solid(line_scan)

    [visApp.update_solid(s) for s in world.solids.values()]
    visApp.update_solid(drone.solid)
    visApp.update_solid(pcd_scan)

    # time.sleep(1)
    keyboard.wait('space')

