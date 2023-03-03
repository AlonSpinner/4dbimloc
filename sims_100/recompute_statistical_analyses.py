from do_statistical_analysis import statistical_analysis
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

for i in range(1,13):
    out = f"out{i}"
    out_folder  = os.path.join(dir_path,out)
    statistical_analysis(out_folder, range(30))