import numpy as np
from Portfolio_Optimization.min_var_port import MinVarPort

def MeanVarPort(bnds, matrix,num_port=10,verbose=True):
    """ Compute the mean variance portfolio for fixed mean which is a quadratic program"""

    lower,upper=bnds
    increment=(upper-lower)/num_port

    meanVector = matrix.mean()
    covMatrix = matrix.cov()
    
    xOptimal = []
    minRiskPoint = []
    expPortfolioReturnPoint = []

    while (lower < upper):

        result = MinVarPort(covMatrix, meanVector, lower, False)
        xOptimal.append(result)
        expPortfolioReturnPoint.append(lower)
        if verbose:
            print("For expected portfolio return of ", lower)
            print("Portfolio Risk: ", np.sqrt(np.dot(np.dot(result, covMatrix), result)))
            print("Portfolio Return: ", np.dot(result, meanVector))
            print("")
        lower = lower + increment
   
    xOptimalArray = np.array(xOptimal)
    minRiskPoint = np.diagonal(np.matmul((np.matmul(xOptimalArray, covMatrix)), np.transpose(xOptimalArray)))
    riskPoint =   np.sqrt(minRiskPoint) 
    retPoint = np.array(expPortfolioReturnPoint) 
    
    return riskPoint, retPoint

def optimize_sharpe(stds,means):
    """ Compute the optimal portfolio for the sharpe ratio given efficient frontier"""
    
    sharpe = means/stds
    max_sharpe_idx = np.argmax(sharpe)

    optimal_mean=means[max_sharpe_idx]
    optimal_std=stds[max_sharpe_idx]
    return optimal_mean,optimal_std