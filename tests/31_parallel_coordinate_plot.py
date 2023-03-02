from bim4loc.evaluation.parallel_coordinate import parallel_coordinate
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    out1_dict = {
                "variation": ["BPFS", "BPFS-t", "BPFS-tg", "logodds"],
                "vel":["1.0", "1.0", "1.0", "1.0"],
                "max_r": [10, 10, 12, 10],
                "weight_calc": ["ros_navigation", "ros_navigation", "ros_navigation", "ros_navigation"],
                "med_traj_error": [0.1, 0.2, 0.3, 0.4]}
    df1 = pd.DataFrame(out1_dict)
    out2_dict = {
                "variation": ["BPFS", "BPFS-t", "BPFS-tg", "logodds"],
                "vel":["1.0", "3.0", "1.0", "1.0"],
                "max_r": [10, 10, 10, 10],
                "weight_calc": ["weight_depended", "weight_depended", "weight_depended", "weight_depended"],
                "med_traj_error": [0.05, 0.15, 0.25, 0.35]}
    df2 = pd.DataFrame(out2_dict)
    out3_dict = {
                "vel":["2.0", "2.0", "2.0", "2.0"],
                "max_r": [10, 10, 10, 10],
                "weight_calc": ["ros_navigation", "ros_navigation", "ros_navigation", "ros_navigation"],
                "variation": ["BPFS", "BPFS-t", "BPFS-tg", "logodds"],
                "med_traj_error": [0.15, 0.25, 0.35, 0.45]}
    df3 = pd.DataFrame(out3_dict)
    df = pd.concat([df1,df2,df3])
    parallel_coordinate(df, "variation",
                        #colors = ['b', 'g', 'r', 'k'],
                        linewidth = 4.0,
                        alpha = 0.3)
    plt.show()




