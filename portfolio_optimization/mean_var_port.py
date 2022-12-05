import numpy as np


from portfolio_optimization.max_var_port import MaxRetPort
from portfolio_optimization.min_var_port import MinVarPort

def MeanVarPort(bnds, matrix):
    """ Compute the mean variance portfolio for fixed mean which is a quadratic program"""

    increment = 0.0001
    lower = bnds[0]
    upper = bnds[1]

    meanVector = matrix.mean()
    covMatrix = matrix.cov()
    
    xOptimal = []
    minRiskPoint = []
    expPortfolioReturnPoint = []

    while (lower < upper):

        result3 = MinVarPort(covMatrix, meanVector, lower, False)
        xOptimal.append(result3.x)
        expPortfolioReturnPoint.append(lower)
        lower = lower + increment
   
    xOptimalArray = np.array(xOptimal)
    minRiskPoint = np.diagonal(np.matmul((np.matmul(xOptimalArray, covMatrix)), np.transpose(xOptimalArray)))
    riskPoint =   np.sqrt(minRiskPoint*251) 
    retPoint = 251*np.array(expPortfolioReturnPoint) 
    
    return riskPoint, retPoint

def optimzie_sharpe(stds,means):
    """ Compute the optimal portfolio for the sharpe ratio given efficient frontier"""
    
    sharpe = means/stds
    max_sharpe_idx = sharpe.idxmax()
    return max_sharpe_idx