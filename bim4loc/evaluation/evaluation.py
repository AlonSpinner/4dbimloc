import numpy as np
import matplotlib.pyplot as plt

def localiztion_error(gt_trajectory,
                      estimated_trajectory,
                      estimated_covariance,
                      dead_reckoning = None):
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
    for i, cov in enumerate(estimated_covariance):
        bounds[i] = np.sqrt(np.sum(np.diag(cov[:3,:3])))
            
    fig, ax = plt.subplots()
    ax.plot(loc_error, label = 'estimation error', color = 'k')
    ax.plot(loc_error + bounds, '--', color = 'k')
    ax.plot(loc_error - bounds, '--', color = 'k')
    
    if dead_reckoning is not None:
        dead_reckoning_error = np.linalg.norm(gt_trajectory[:,:3] - dead_reckoning[:,:3], axis=1)
        ax.plot(dead_reckoning_error, label='dead reckoning error', color='r')
    
    ax.set_title('Localization Error')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error')
    ax.grid(True)

def map_entropy(perfect_beliefs,estimated_beliefs):
    """
    Args:
        perfect_beliefs - np.array of shape (n, n_elements), assuming perfect trajectory
        estimated_beliefs - np.array of shape (n, n_elements)
    Returns:
        plots map entropy
    """
    perfect = -np.sum(perfect_beliefs * np.log(perfect_beliefs), axis=1)
    estimated = -np.sum(estimated_beliefs * np.log(estimated_beliefs), axis=1)

    fig, ax = plt.subplots()
    ax.plot(perfect, label='perfect')
    ax.plot(estimated, label='estimated')
    ax.set_title('Map Entropy')
    ax.set_xlabel('Time')
    ax.set_ylabel('Entropy')
    return 