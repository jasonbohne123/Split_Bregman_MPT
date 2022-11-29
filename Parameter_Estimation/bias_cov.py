import numpy as np

def reg_cov(cov,penalty=0.5):
    
    cov = matrix.cov()

    # Compute sample covariance matrix
    sampleCov = np.dot(returns.values.T, returns.values) / len(matrix.columns)
    sampleMean = matrix.mean()
    
    I = np.identity(len(cov.columns))
    
    # Compute the estimates
    mu = np.dot(cov, I)
    alpha = np.linalg.norm(cov - mu * I)
    beta = np.linalg.norm(sampleCov - cov)
    delta = np.linalg.norm(sampleCov - mu * I)

    row1 = (beta**2)/(alpha**2 + beta**2)
    row2 = (alpha**2)/(delta**2)
    
    covEstimate = row1 * mu * I + row2 * sampleCov  
    
    #(1-penalty)*cov+penalty*np.identity(cov.shape[0])
    return covEstimate