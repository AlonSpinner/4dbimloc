import open3d as o3d

def key_callback(vis, key):
    if key == ord('q'):
        exit()
    else:
        print("Key pressed:", chr(key))

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.register_key_callback(key_callback)
vis.run()