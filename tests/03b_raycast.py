import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, PcdSolid, LinesSolid
from bim4loc.agents import Drone
from bim4loc.sensors import Lidar1D
from bim4loc.maps import RayCastingMap
from bim4loc.geometry.raycaster import NO_HIT
import time
import keyboard

solids = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = Pose2z(3,3,0,1.5))
# sensor = Lidar1D(angles = np.linspace(-np.pi/2,np.pi/2,3))
sensor = Lidar1D()
sensor.std = 0.05; sensor.piercing = True; sensor.max_range = 1000.0
drone.mount_sensor(sensor)
world = RayCastingMap(solids)

straight = Pose2z(0.5,0,0,0)
turn_left = Pose2z(0,0,np.pi/8,0)
turn_right = Pose2z(0,0,-np.pi/8,0)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

visApp = VisApp()
for s in solids:
    visApp.add_solid(s,"world")
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

visApp.add_solid(drone.solid)
pcd_scan = PcdSolid()
# pcd_scan.material.point_size = 5.0
visApp.add_solid(pcd_scan)
line_scan = LinesSolid()
visApp.add_solid(line_scan)

time.sleep(1)
for a in actions:
    drone.move(a)
    z, z_ids = sensor.sense(drone.pose, world, n_hits = 20)
    
    z_flat = np.array([])
    angles_flat = np.array([])
    z_ids_flat = []
    for zi,ai,zidi in zip(z, sensor.angles, z_ids):
        zi_valid = zi[zi < sensor.max_range]
        z_flat = np.hstack((z_flat, zi_valid))
        angles_flat = np.hstack((angles_flat,np.full_like(zi_valid, ai)))
        [z_ids_flat.append(_) for _ in zidi if _ != NO_HIT]

    drone_p = np.vstack((z_flat * np.cos(angles_flat), 
            z_flat * np.sin(angles_flat),
            np.zeros_like(z_flat)))
    p = drone.pose.transform_from(drone_p)
    
    for s in world.solids.values():
        if s.iguid in z_ids_flat:
            s.material.base_color = (1,0,0,1)
        else:
            s.material.base_color = np.hstack((s.ifc_color,1))
    pcd_scan.update(p.T)
    p = np.hstack((drone.pose.t, p))
    line_ids = np.zeros((p.shape[1],2), dtype = int)
    line_ids[:,1] = np.arange(p.shape[1])
    line_scan.update(p.T, line_ids)

    [visApp.update_solid(s) for s in world.solids.values()]
    visApp.update_solid(drone.solid)
    visApp.update_solid(pcd_scan)
    visApp.update_solid(line_scan)

    # time.sleep(1)
    keyboard.wait('space')
