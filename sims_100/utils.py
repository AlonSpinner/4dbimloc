import os

def create_folders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    os.mkdir(os.path.join(folder_path, "data"))
    os.mkdir(os.path.join(folder_path, "results"))
    os.mkdir(os.path.join(folder_path, "media"))
    os.mkdir(os.path.join(folder_path, "statistical_analysis"))

