import numpy as np

def reg_mean(matrix, days):
    """ Regularized James- Stein mean vector estimator
    """
    
    cov = matrix.cov()

    # Compute sample covariance matrix
    sampleCov = np.dot(matrix.values.T, matrix.values) / len(matrix.columns)
    sampleMean = matrix.mean()
    
    # Find the mean estimate
    eta = max(sampleMean.sum() / sampleMean.size, 0.0004)   
    denom = np.dot(np.transpose((sampleMean - eta*np.ones(shape = (len(sampleMean),))).values.reshape((len(sampleMean), 1))),
                   np.dot(np.linalg.pinv(covEstimate), (sampleMean - eta*np.ones(shape = (len(sampleMean),))).values.reshape((len(sampleMean), 1))))

    rightside = ((len(sampleMean) - 2) / days) / (denom + len(sampleMean) + 2)
    leftside = 1
    rho = min(leftside, rightside)[0][0]       
    meanEstimate = (1 - rho)*sampleMean + rho*eta*np.ones(shape = (len(sampleMean),))
    
    return meanEstimate