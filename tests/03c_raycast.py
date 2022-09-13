import numpy as np
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, ScanSolid
from bim4loc.agents import Drone
from bim4loc.sensors import Lidar
from bim4loc.maps import RayCastingMap
import time
import keyboard

full_solids = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = np.array([3.0, 3.0, 1.5, 0.0]))
sensor = Lidar(angles_u = np.array([0.2]), angles_v = np.array([0.0]))
sensor.piercing = False
sensor.max_range = 1000.0
drone.mount_sensor(sensor)

solids = [s for s in full_solids if s.name == "1UH7XjeubFPe8ud33kpdAD"]# or s.name == "22fuoCLrXEA9lnNzqjOo6F"] 
world = RayCastingMap(solids)

visApp = VisApp()
for s in world.solids:
    visApp.add_solid(s,"world")
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

visApp.add_O3DVisualizer('o3d_window','o3d_scene')
for s in world.solids:
        visApp.O3DVis_add_solid(s,"o3d_scene")
visApp.redraw('o3d_scene')
visApp.O3DVis_reset_camera('o3d_window')
visApp.O3DVis_show_axes('o3d_window')

visApp.add_solid(drone.solid)
vis_scan = ScanSolid("scan")
visApp.add_solid(vis_scan)


z, z_solid_names, z_normals, p = drone.scan(world, project_scan = True)

#show rays
vis_scan.update(drone.pose[:3], p.T)
visApp.update_solid(vis_scan)
[visApp.update_solid(s) for s in world.solids]
visApp.update_solid(drone.solid)

