import numpy as np
from bim4loc.geometry.poses import Pose2z
from bim4loc.binaries.paths import IFC_ONLY_WALLS_PATH
from bim4loc.visualizer import VisApp
from bim4loc.solids import ifc_converter, PcdSolid
from bim4loc.agents import Drone
from bim4loc.sensors import Lidar1D
from bim4loc.maps import RayTracingMap
import bim4loc.geometry.raytracer as myRayTracer
import time

objects = ifc_converter(IFC_ONLY_WALLS_PATH)
drone = Drone(pose = Pose2z(3,3,0,1.5))
sensor = Lidar1D(); sensor.std = 0.05
drone.mount_sensor(sensor)
world = RayTracingMap(objects)

rayTracingScene = myRayTracer.map2scene(world)

straight = Pose2z(0.5,0,0,0)
turn_left = Pose2z(0,0,np.pi/8,0)
turn_right = Pose2z(0,0,-np.pi/8,0)
actions = [straight] * 9 + [turn_left] * 4 + [straight] * 8 + [turn_right] * 4 + [straight] * 20

visApp = VisApp()
for o in objects:
    visApp.add_solid(o,"world")
visApp.redraw()
visApp.setup_default_camera("world")
visApp.show_axes()

visApp.add_solid(drone.solid)
pcd_scan = PcdSolid()
visApp.add_solid(pcd_scan)

time.sleep(1)
for i, a in enumerate(actions):
    if i == 1:
        start_time = time.time()

    drone.move(a)
    rays = sensor.get_rays(drone.pose)
    z_values, z_ids = myRayTracer.raytrace(rays, *rayTracingScene)
    z_values, z_names = myRayTracer.post_process_raytrace(z_values, z_ids, world.solid_names, n_hits = 1)

    p = sensor.project_scan(drone.pose, np.array(z_values)[:,0])
    # z, _, p = drone.scan(world)
    pcd_scan.update(p.T)

    visApp.update_solid(drone.solid)
    visApp.update_solid(pcd_scan)

    time.sleep(0.01)

end_time = time.time()
print(f'finished in {end_time - start_time} seconds')
myRayTracer.raytrace.parallel_diagnostics(level = 1)
