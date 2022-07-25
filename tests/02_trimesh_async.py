from bim4loc.binaries.paths import IFC_SIMPLE_PATH, DRONE_PATH, IFC_TEST_PATH
from bim4loc.ifc import converter
import trimesh
import numpy as np
import asyncio

# trimesh does not support animations. so it doesn't matter that we run it in a different thread 

async def showScene(scene):
        scene.show()

async def main():
    objects = converter(IFC_TEST_PATH)

    s = trimesh.scene.Scene()
    for o in objects:
        g = o.geometry
        v = np.asarray(g.vertices)
        f = np.asarray(g.triangles)
        tm = trimesh.Trimesh(vertices=v, faces=f)
        s.add_geometry(tm)

    task =  asyncio.create_task(showScene(s))

    print('yes')

asyncio.run(main())


