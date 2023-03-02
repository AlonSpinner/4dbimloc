import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

def min_jerk(x0,x1,y0,y1, N = 100):
    dx = x1-x0
    x = np.linspace(x0,x1,N)
    tau = (x - x0)/dx
    y = y0 + (y1-y0)*(10*tau**3 - 15*tau**4 + 6*tau**5)
    return x,y

def parallel_coordinate(df : pd.DataFrame,
                        color_variable : str,
                        linewidth = 2.0):
    #normalize data and put it in a parallel df
    ndf = df.copy()
    #check which variables are numeric - thank gpt chat for this stupidness
    are_numeric = {}
    for variable, values in df.items():
        are_numeric[variable] = np.issubdtype(values,np.number)

    non_numeric_variables_maps = {}
    for variable, is_numeric in are_numeric.items():
        if is_numeric is False: #need to enumerate
            unqiue_set = np.unique(df[variable].values)
            assigned_values = np.linspace(0.0,1.0,len(unqiue_set))
            non_numeric_variables_maps[variable] = dict(zip(unqiue_set,assigned_values))
            for float_value, string_value in zip(assigned_values, unqiue_set):
                ndf[variable].replace(string_value, float_value, inplace=True)
        else:
            max_val = max(df[variable])
            min_val = min(df[variable])
            if min_val == max_val:
                ndf[variable] = 0.5
            else:
                ndf[variable] = (df[variable] - min_val)/(max_val - min_val)
 
    #plot
    fig = plt.figure(figsize = (16,10))
    ax = fig.add_subplot(111)
    variables = df.columns.to_list()
    xticks = np.arange(len(variables))/(len(variables)-1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(variables)
    ax.set_xlim(0.0,1.0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis = 'x', labelrotation = 45)

    color_variable_index = df.columns.get_loc(color_variable)
    cmap = plt.get_cmap('viridis')
    
    for row in ndf.values:
        for i in range(len(variables)-1):
            x,y = min_jerk(xticks[i], xticks[i+1], row[i], row[i+1])
            color = cmap(row[color_variable_index])
            ax.plot(x,y, linewidth = linewidth, color = color)

    #add ylims
    for i in range(len(variables)):
        twinx = ax.twinx()
        twinx.spines['right'].set_position(("axes", xticks[i]))
        twinx.spines['top'].set_visible(False)
        twinx.spines['bottom'].set_visible(False)
        twinx.spines['left'].set_visible(False)
        yticks = twinx.get_yticks()
        twinx.yaxis.set_major_locator(plt.FixedLocator(yticks))
        if are_numeric[variables[i]]:
            max_val = df[variables[i]].max()
            min_val = df[variables[i]].min()
            if min_val == max_val:
               twinx.set_yticks([0.5])
               twinx.set_yticklabels([str(max_val)])
            else:
                unormalize_yticks = (yticks*(max_val - min_val)) + min_val
                twinx.set_yticklabels(unormalize_yticks)
                twinx.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            variable_map = non_numeric_variables_maps[variables[i]]
            twinx.set_yticks(np.array(list(variable_map.values())))
            twinx.set_yticklabels(variable_map.keys())

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
parallel_coordinate(df, "variation")
plt.show()




