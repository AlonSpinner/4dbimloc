import os
import sys

if __name__ == "__main__":
    folder_path = sys.argv[1]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    os.mkdir(os.path.join(folder_path, "data"))
    os.mkdir(os.path.join(folder_path, "results"))
    os.mkdir(os.path.join(folder_path, "media"))
    os.mkdir(os.path.join(folder_path, "statistical_analysis"))

