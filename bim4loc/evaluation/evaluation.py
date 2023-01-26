import numpy as np
import matplotlib.pyplot as plt
from bim4loc.random.utils import compute_entropy, compute_cross_entropy

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