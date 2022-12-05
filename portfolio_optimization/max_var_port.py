import numpy as np
from scipy import optimize

def MaxRetPort(mean):
    """ Compute the maximum return portfolio which is just a linear program"""
    
    c = (np.multiply(-1, mean))
    A = np.ones([len(mean), 1]).T
    b = [1]
    res = optimize.linprog(c, A_ub = A, b_ub = b, bounds = (0, 0.33), method = 'simplex') 
    
    return res.x