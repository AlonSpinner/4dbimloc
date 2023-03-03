import os
from bim4loc.evaluation.parallel_coordinate import parallel_coordinate
from bim4loc.utils.load_yaml import load_parameters
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
data = {}
variation_enumeration = {1 : "BPFS", 2 : "BPFS-t", 3 : "BPFS-tg", 4 : "logodds"}

for i in range(1,13):
    out = f"out{i}"
    out_folder  = os.path.join(dir_path,out)
    yaml_file_path = os.path.join(out_folder,
                                  "parameters.yaml")
    stats_file_path = os.path.join(out_folder,
                                   "statistical_analysis",
                                    "statistical_data.p")

    parameters = load_parameters(yaml_file_path)
    stats = pickle.Unpickler(open(stats_file_path, "rb")).load()

    data[out] = {'parameters': parameters, 'stats': stats}

def get_tuned_params(params):
    tuned_params = {}
    tuned_params["sensor_max_range"] = params["sensor_max_range"]
    tuned_params["velocity_factor"] = params["velocity_factor"]
    tuned_params["rbpf_weight_calculation_method"] = params["rbpf_weight_calculation_method"]
    return tuned_params

def get_df(medians = False, variations = [1,2,3,4], outs = None):
    df = pd.DataFrame()
    row_index = 0
    for key, value in data.items():
        if outs is not None:
            if key not in outs:
                continue
        tuned_params = get_tuned_params(value["parameters"])
        if tuned_params['rbpf_weight_calculation_method'] == "default":
           tuned_params['rbpf_weight_calculation_method'] = "[14]" 
        else:
            tuned_params['rbpf_weight_calculation_method'] = "ours"
        for i in variations:
            mean_traj_errors = value["stats"]["mean_traj_err"][i]
            terminal_cross_entropy_errors = value["stats"]["final_mean_ce_err"][i]
            termnial_accuries = value["stats"]["final_acc"][i]
            if medians is False:
                for m_l_err, t_c_e, t_acc in zip(mean_traj_errors, terminal_cross_entropy_errors, termnial_accuries):
                    row_dict = {"variation" : variation_enumeration[i]} \
                            | tuned_params \
                            | {"mean localization error" : m_l_err} \
                            | {"terminal cross entropy error" : t_c_e} \
                            | {"terminal accuracy" : t_acc}
                    df = pd.concat([df,pd.DataFrame(row_dict, index = [row_index])])
                    row_index += 1
            else:
                row_dict = {"variation" : variation_enumeration[i]} \
                    | tuned_params \
                    | {"mean localization error" : np.median(mean_traj_errors)} \
                    | {"terminal cross entropy error" : np.median(terminal_cross_entropy_errors)} \
                    | {"terminal accuracy" : np.median(termnial_accuries)}
                df = pd.concat([df,pd.DataFrame(row_dict, index = [row_index])])
                row_index += 1

    df.rename(columns = {"sensor_max_range": "sensor max range"},inplace=True) 
    df.rename(columns = {"rbpf_weight_calculation_method": "weight calculation method"},inplace=True) 
    df.rename(columns = {"velocity_factor": "velocity factor"},inplace=True) 
    return df

def mega_table(only14 = False):
    '''
    we concluded that the new reweighting scheme does not help.

    before running, make sure that values are calculated on best particle 
    and not expected value

    mega_table STRUCTURE
                BPFS      BPFS-t   BPFS-tg   logodds
               ours,14   ours,14   ours,14   ours,14
       mr = 10
    VF = 1.0 
       mr =  6

       mr = 10
    VF = 2.0 
       mr =  6

       mr = 10
    VF = 4.0 
       mr =  6
    '''
    df = get_df(medians = True)
    variations = ['BPFS', 'BPFS-t', 'BPFS-tg', 'logodds']
    velocity_factor_super_rows = [1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    sensor_max_range_sub_rows = [6.0, 10.0, 6.0, 10.0, 6.0, 10.0]
    weight_calc_sub_cols = ["ours", "[14]"]
    N_rows = 6
    le_rows = []; ce_rows = []; acc_rows = []
    for i in range(N_rows): 
        le_row = []; ce_row = []; acc_row = []
        for v in variations:
            for w in weight_calc_sub_cols:
                q = (df["variation"] == v) & \
                (df["velocity factor"] == velocity_factor_super_rows[i]) & \
                (df["sensor max range"] == sensor_max_range_sub_rows[i]) & \
                (df["weight calculation method"] == w)
                filtered_df = df[q]
                values = filtered_df.loc[:,["mean localization error",
                                             "terminal cross entropy error",
                                             "terminal accuracy"]].values
                values = np.around(values, decimals = 2)
                le_row.append(values[0,0]); ce_row.append(values[0,1]); acc_row.append(values[0,2])
        le_rows.append(le_row); ce_rows.append(ce_row); acc_rows.append(acc_row)

    le_rows = np.array(le_rows); ce_rows = np.array(ce_rows); acc_rows = np.array(acc_rows)                
    
    if only14:
        le_rows = le_rows[:,1::2]
        ce_rows = ce_rows[:,1::2]
        acc_rows = acc_rows[:, 1::2]

    print("mean localization error")
    print(le_rows)
    print("terminal cross entropy error")
    print(ce_rows)
    print("terminal accuracy")
    print(acc_rows)

# def box_plots():

if __name__ == "__main__":
    df = get_df(medians = False)
    parallel_coordinate(df, "variation", colors = ['b', 'g', 'r', 'k'],
                         linewidth = 1.3, alpha = 0.05)
    
    # df = get_df(medians = False, variations = [1])
    # parallel_coordinate(df, "weight calculation method", 
    #                      linewidth = 1.3, alpha = 0.05)

    # mega_table(only14 = True)
    plt.show()