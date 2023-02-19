import numpy as np
import matplotlib.pyplot as plt
from bim4loc.random.utils import compute_entropy, compute_cross_entropy, hamming_distance, jaccard_distance

def localiztion_error(gt_trajectory,
                      estimated_trajectory,
                      estimated_covariance = None,
                      dead_reckoning = None,
                      plot = False):
    """Compute the localization error.
    Args:
        gt_trajectory - np.array of shape (n, 4)
        estimated_trajectory - np.array of shape (n, 4)
        estimated_covariance - np.array of shape (n, 4, 4)
        dead_reckoing - np.array of shape (n, 4)
    Returns:
        plots localization error
    """
    loc_error = np.linalg.norm(gt_trajectory[:,:3] - estimated_trajectory[:,:3], axis=1)
    bounds = np.zeros_like(loc_error)
    if estimated_covariance is not None:
        for i, cov in enumerate(estimated_covariance):
            bounds[i] = np.sqrt(np.sum(np.diag(cov[:3,:3])))
        
    if dead_reckoning is not None:
        dead_reckoning_error = np.linalg.norm(gt_trajectory[:,:3] - dead_reckoning[:,:3], axis=1)
    
    if plot is True:

        fig, ax = plt.subplots()
        ax.plot(loc_error, label = 'estimation error', color = 'k')
        if estimated_covariance is not None:
            ax.plot(loc_error + bounds, '--', color = 'k')
            ax.plot(loc_error - bounds, '--', color = 'k')
        if dead_reckoning is not None:
            ax.plot(dead_reckoning_error, label='dead reckoning error', color='r')

        ax.set_title('Localization Error')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error')
        ax.grid(True)
        ax.legend()

    return loc_error, bounds

def map_entropy(estimated_beliefs, perfect_beliefs = None, plot = False):
    """
    Args:
        perfect_beliefs - np.array of shape (n, n_elements), assuming perfect trajectory
        estimated_beliefs - np.array of shape (n, n_elements)
    Returns:
        plots map entropy
    """
    estimated = compute_entropy(estimated_beliefs)

    perfect = np.zeros_like(estimated)
    if perfect_beliefs is not None:
                perfect = compute_entropy(perfect_beliefs)

    if plot is True:
        fig, ax = plt.subplots()
        ax.plot(estimated, label='rbpf')
        if perfect is not None:
            ax.plot(perfect, label='mapping with known poses')
        ax.set_title('Map Entropy')
        ax.set_xlabel('Time')
        ax.set_ylabel('Entropy')
        ax.grid(True)
        ax.legend()

    return estimated,perfect

def cross_entropy_error(ground_truth, estimated_beliefs, perfect_beliefs = None, plot = False):
    '''
    perfect_beliefs - np.array of shape (n, n_elements), assuming perfect trajectory
    estimated_beliefs - np.array of shape (n, n_elements)
    ground_truth - np.array of shape (n_elements)
    '''
    ground_truth = np.tile(ground_truth, (estimated_beliefs.shape[0], 1))
    estimated = compute_cross_entropy(ground_truth, estimated_beliefs)

    perfect = np.zeros_like(estimated)
    if perfect_beliefs is not None:
                perfect = compute_cross_entropy(ground_truth, perfect_beliefs)
    
    if plot is True:
        fig, ax = plt.subplots()
        ax.plot(estimated, label='rbpf')
        if perfect_beliefs is not None:
                ax.plot(perfect, label='mapping with known poses')
        ax.set_title('Cross Entropy Error')
        ax.set_xlabel('Time')
        ax.set_ylabel('Entropy')
        ax.grid(True)
        ax.legend()

    return estimated, perfect

def belief_map_accuracy(ground_truth, estimated_beliefs, perfect_beliefs = None, plot = False):
    '''
    perfect_beliefs - np.array of shape (n, n_elements), assuming perfect trajectory
    estimated_beliefs - np.array of shape (n, n_elements)
    ground_truth - np.array of shape (n_elements)
    '''
    Ngt = len(ground_truth)
    binary_estimated_beleifs = estimated_beliefs > 0.9
    True_Positives = np.zeros(estimated_beliefs.shape[0])
    True_Negatives = np.zeros(estimated_beliefs.shape[0])
    for i in range(estimated_beliefs.shape[0]):
        True_Positives[i] = np.sum(binary_estimated_beleifs[i][:Ngt] * ground_truth)
        True_Negatives[i] = np.sum((1-binary_estimated_beleifs[i][:Ngt]) * (1-ground_truth))
    estimated = (True_Positives + True_Negatives)/ len(ground_truth)

    perfect = np.zeros_like(estimated)
    if perfect_beliefs is not None:
        binary_perfect_beliefs = perfect_beliefs > 0.9
        True_Positives = np.zeros(perfect_beliefs.shape[0])
        True_Negatives = np.zeros(perfect_beliefs.shape[0])
        for i in range(binary_perfect_beliefs.shape[0]):
            True_Positives[i] = np.sum(binary_perfect_beliefs[i][:Ngt] * ground_truth)
            True_Negatives[i] = np.sum((1-binary_perfect_beliefs[i][:Ngt]) * (1-ground_truth))
        perfect = (True_Positives + True_Negatives)/ len(ground_truth)
    
    if plot is True:
        fig, ax = plt.subplots()
        ax.plot(estimated, label='rbpf')
        if perfect_beliefs is not None:
                ax.plot(perfect, label='mapping with known poses')
        ax.set_title('Cross Entropy Error')
        ax.set_xlabel('Time')
        ax.set_ylabel('Entropy')
        ax.grid(True)
        ax.legend()

    return estimated, perfect

def percentile_boxes_right(expected_belief_map, ground_truth_beliefs, 
                            gt_electric_boxes_names, gt_electric_boxes_indicies,
                            sim_electric_boxes_indicies, sim_electric_boxes_names,
                            gt_electric_boxes_seen_counter):
    #seen ground truth boxes
    seen_boxes_names = []
    seen_boxes_indicies = []
    for (box_name, seen_counter) in gt_electric_boxes_seen_counter.items():
        if seen_counter > 20:
            seen_boxes_names.append(box_name)
            seen_boxes_indicies.append(gt_electric_boxes_indicies[gt_electric_boxes_names.index(box_name)])

    if len(seen_boxes_names) == 0:
        return 0, 0

    N_boxes_got_right = 0
    for seen_box_name, seen_box_index in zip(seen_boxes_names, seen_boxes_indicies):
        #collect indicies of all boxes that start with box_name
        relevant_boxes_indicies = []
        for j,sim_box_name in enumerate(sim_electric_boxes_names):
            if sim_box_name.startswith(seen_box_name):
                relevant_boxes_indicies.append(sim_electric_boxes_indicies[j])
        
        best_box_index = relevant_boxes_indicies[np.argmax(expected_belief_map[-1][relevant_boxes_indicies])]
        if best_box_index != seen_box_index:
            continue #not worth nitty picking. we chose wrong box
        is_accurate = True
        for index in relevant_boxes_indicies:
            if index != best_box_index: #index points to false variation, should be low belief
                if expected_belief_map[-1][index] > 0.1:
                    is_accurate = False
            if index == best_box_index: #index points to high best variation, should be high belief
                if expected_belief_map[-1][index] < 0.9:
                    is_accurate = False
        if is_accurate:
            N_boxes_got_right += 1

    N_seen_boxes = len(seen_boxes_names)
    return N_boxes_got_right, N_seen_boxes

def maps_average_distance(v : np.ndarray, distance_metric = 'hamming'):
    '''
    v - binary matrix of size mxn, with m maps 
    '''
    total_distance = 0
    m = v.shape[0]
    min_distances = np.zeros(m)
    for i in range(m):
        di_min = np.inf
        for j in range(m):
            if distance_metric == 'hamming':
                dij = hamming_distance(v[i],v[j])
            elif distance_metric == 'jaccard':
                dij = jaccard_distance(v[i],v[j])
            else:
                raise Exception("wrong method name")            
            if i != j and dij < di_min:
                di_min = dij        

            if j > i:
                total_distance  += dij
        min_distances[i] = di_min
    
    average_distance = total_distance / (m * (m-1)/2)

    return average_distance, min_distances

