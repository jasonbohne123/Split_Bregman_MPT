import numpy as np
from scipy import optimize


def MinVarPort(cov, meanVector = None, fixed_return = None, flag = True,max_pos=0.33):
    """ Compute the minimum variance portfolio which is a quadratic problem 
    """
    
    # Objective function
    def f(weights, cov):
        return np.dot(np.dot(weights, cov), weights)
    
    # Constraints
    def wConstraint(weights):
        A = np.ones(weights.shape)
        b = 1
        val = np.dot(weights, A) - b        
        return val

    def fix_return_Constraint(weights, meanVector, fixed_return):
        A = np.array(meanVector)
        b = fixed_return
        val = np.dot(weights, A) - b
        return val
    
    # Initialize weights and constraints
    weights = np.repeat(1, len(cov))
    
    # Assume we don't want a single position larger than 33%
    bnds = tuple([(0, max_pos) for _ in weights])
    
    # if we want to compute min var
    if (flag == True):
        cons = ({'type': 'eq', 'fun' : wConstraint})
        opt = optimize.minimize (f, x0 = weights, args = (cov),  
                                 bounds = bnds, constraints = cons, tol = 10**-3,options={'maxiter':100})
    
    # for a fixed porfolio return 
    else:
        cons = ({'type': 'eq', 'fun': wConstraint}, {'type': 'ineq', 'fun': fix_return_Constraint, 'args': (meanVector, fixed_return)})
        opt = optimize.minimize (f, args = (cov), method ='trust-constr', 
                                 x0 = weights, bounds = bnds, constraints = cons, tol = 10**-3,options={'maxiter':100})
    
    return opt.x