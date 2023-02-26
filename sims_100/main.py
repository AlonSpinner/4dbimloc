from data_collection import create_data
from run_simulations import run_simulation
import numpy as np
U_COV = np.diag([0.2, 0.1, 1e-25, np.radians(1)])**2
for i in range(0,1):
    create_data(i, U_COV, "data", vis_on = False)
    run_simulation(i, "data" ,"results", vis_on = False)
    print(f'finished {i}')
print('done')
