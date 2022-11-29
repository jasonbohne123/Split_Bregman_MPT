import numpy as np

def reg_mean(matrix, days):
    
    cov = matrix.cov()

    # Compute sample covariance matrix
    sampleCov = np.dot(matrix.values.T, matrix.values) / len(matrix.columns)
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
    
    # Find the mean estimate
    eta = max(sampleMean.sum() / sampleMean.size, 0.0004)   
    denom = np.dot(np.transpose((sampleMean - eta*np.ones(shape = (len(sampleMean),))).values.reshape((len(sampleMean), 1))),
                   np.dot(np.linalg.pinv(covEstimate), (sampleMean - eta*np.ones(shape = (len(sampleMean),))).values.reshape((len(sampleMean), 1))))

    rightside = ((len(sampleMean) - 2) / days) / (denom + len(sampleMean) + 2)
    leftside = 1
    rho = min(leftside, rightside)[0][0]       
    meanEstimate = (1 - rho)*sampleMean + rho*eta*np.ones(shape = (len(sampleMean),))
    
    return meanEstimate