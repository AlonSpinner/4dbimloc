#based on: https://stackoverflow.com/questions/58666635/implementing-pca-with-numpy
import numpy as np


def pca(X, n_components=2):
    """
    Assumes observations in X are passed as rows of a numpy array.
    """

    # Translate the dataset so it's centered around 0
    translated_X = X - np.mean(X, axis=0)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    e_values, e_vectors = np.linalg.eigh(np.cov(translated_X.T))

    # Sort eigenvalues and their eigenvectors in descending order
    e_ind_order = np.flip(e_values.argsort())
    e_values = e_values[e_ind_order]
    e_vectors = e_vectors[e_ind_order]

    # Save the first n_components eigenvectors as principal components
    principal_components = np.take(e_vectors, np.arange(n_components), axis=0)

    transform = principal_components.T
    X_hat = translated_X @ transform
    return X_hat, transform