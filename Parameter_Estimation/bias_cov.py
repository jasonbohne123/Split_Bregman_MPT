import numpy as np

def reg_cov(matrix):
    """ Regularized L2 Covariance Matrix estimator
    """
    cov = matrix.cov()

    # Compute sample covariance matrix
    sampleCov = np.dot(matrix.values.T, matrix.values) / len(matrix.columns)
    I = np.identity(len(cov.columns))
    
    # Compute the estimates
    mu = np.dot(cov, I)
    alpha = np.linalg.norm(cov - mu * I)
    beta = np.linalg.norm(sampleCov - cov)
    delta = np.linalg.norm(sampleCov - mu * I)

    row1 = (beta**2)/(alpha**2 + beta**2)
    row2 = (alpha**2)/(delta**2)
    
    covEstimate = row1 * mu * I + row2 * sampleCov  
    
    return covEstimate