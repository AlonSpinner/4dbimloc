import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.transforms

def find_key_by_index(d : dict, value : int):
    for key, val in d.items():
        if val == value:
            return key
    raise ValueError(f'Value {value} not found in dict')

def min_jerk(x0,x1,y0,y1, N = 100):
    dx = x1-x0
    x = np.linspace(x0,x1,N)
    tau = (x - x0)/dx
    y = y0 + (y1-y0)*(10*tau**3 - 15*tau**4 + 6*tau**5)
    return x,y

def parallel_coordinate(df : pd.DataFrame,
                        color_variable : str,
                        linewidth = 2.0,
                        alpha = 0.5,
                        label_dx_shift = -5/72,
                        label_dy_shift = 10/72,
                        colors = None,
                        axbox = [0.1, 0.2, 0.8, 0.6],
                        x_label_size = 12,
                        x_label_rotation = -0,
                        y_label_size = 11,
                        y_decimel_round = 2):
    are_numeric = {}
    for variable, values in df.items():
        are_numeric[variable] = np.issubdtype(values,np.number)
    if colors is not None:
        assert len(colors) == len(np.unique(df[color_variable].values))
        assert are_numeric[color_variable] is False
        color_map = dict(zip(np.unique(df[color_variable].values), colors))
    else:
        cmap = plt.get_cmap('plasma')

    #normalize data and put it in a parallel df
    ndf = df.copy()

    non_numeric_variables_maps = {}
    for variable, is_numeric in are_numeric.items():
        if is_numeric is False: #need to enumerate
            unquie_set = np.unique(df[variable].values)
            if len(unquie_set) == 0:
                assigned_values = np.array([0.5])
            elif len(unquie_set) == 2:
                assigned_values = np.array([0.25, 0.75])
            else:
                assigned_values = np.linspace(0.0,1.0,len(unquie_set))
            non_numeric_variables_maps[variable] = dict(zip(unquie_set,assigned_values))
            for float_value, string_value in zip(assigned_values, unquie_set):
                ndf[variable].replace(string_value, float_value, inplace=True)
        else: #if it is numeric
            max_val = max(df[variable])
            min_val = min(df[variable])
            if min_val == max_val:
                ndf[variable] = 0.5
            else:
                ndf[variable] = (df[variable] - min_val)/(max_val - min_val)
 
    #plot
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_axes(axbox)
    variables = df.columns.to_list()
    xticks = np.arange(len(variables))/(len(variables)-1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(variables)
    ax.set_xlim(0.0,1.0)
    ax.set_ylim(0.0,1.0) #important as twinx will be set for the same
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelrotation = x_label_rotation,
                               labelsize = x_label_size)
    ax.yaxis.set_visible(False)

    color_variable_index = df.columns.get_loc(color_variable)
    for row in ndf.values:
        for i in range(len(variables)-1):
            x,y = min_jerk(xticks[i], xticks[i+1], row[i], row[i+1])
            if colors is None:
                color = cmap(row[color_variable_index])
            else:
                #inefficient but whatever
                key = find_key_by_index(non_numeric_variables_maps[variables[color_variable_index]],
                                        row[color_variable_index])
                color = color_map[key]
            ax.plot(x,y, linewidth = linewidth, color = color,alpha = alpha)

    #add ylims
    for i in range(len(variables)):
        twinx = ax.twinx()
        twinx.spines['right'].set_position(("axes", xticks[i]))
        twinx.spines['top'].set_visible(False)
        twinx.spines['bottom'].set_visible(False)
        twinx.spines['left'].set_visible(False)
        yticks = twinx.get_yticks()
        twinx.yaxis.set_major_locator(plt.FixedLocator(yticks))
        twinx.set_ylim(0,1)
        #set y ticks and their labels
        twinx.tick_params(axis = 'y',labelsize = y_label_size)
        if are_numeric[variables[i]]:
            max_val = df[variables[i]].max()
            min_val = df[variables[i]].min()
            if min_val == max_val:
               twinx.set_yticks([0.5])
               twinx.set_yticklabels([str(max_val)])
            else:
                unormalize_yticks = np.around((yticks*(max_val - min_val)) + min_val,y_decimel_round)
                twinx.set_yticklabels(unormalize_yticks)
        else:
            variable_map = non_numeric_variables_maps[variables[i]]
            twinx.set_yticks(np.array(list(variable_map.values())))
            twinx.set_yticklabels(variable_map.keys())

        #https://stackoverflow.com/questions/28615887/how-to-move-a-tick-label-in-matplotlib
        offset = matplotlib.transforms.ScaledTranslation(label_dx_shift,
                                                         label_dy_shift,
                                                         fig.dpi_scale_trans)
        # apply offset transform to all x ticklabels.
        for label in twinx.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

if __name__ == "__main__":
    out1_dict = {
                "variation": ["BPFS", "BPFS-t", "BPFS-tg", "logodds"],
                "vel":["1.0", "1.0", "1.0", "1.0"],
                "max_r": [10, 10, 10, 10],
                "weight_calc": ["ros_navigation", "ros_navigation", "ros_navigation", "ros_navigation"],
                "med_traj_error": [0.1, 0.2, 0.3, 0.4]}
    df1 = pd.DataFrame(out1_dict)
    out2_dict = {
                "variation": ["BPFS", "BPFS-t", "BPFS-tg", "logodds"],
                "vel":["1.0", "1.0", "1.0", "1.0"],
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
    parallel_coordinate(df, "med_traj_error",
                        #colors = ['b', 'g', 'r', 'k'],
                        linewidth = 4.0,
                        alpha = 0.3)
    plt.show()




