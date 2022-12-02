import numpy as np
from Split_Bregman.split_bregman_opt import split_bregman

def grid_search(params,returns_mean,returns_cov,tol=1e-8,maxiter=25):
    """ Grid Search for optimal lambda1 and lambda2 under constrained portfolio problem 
    """
    results={}
    for i in params:
        lambda1,lambda2=i
        lb,ub=-1,1
        w,error,status=split_bregman(returns_cov,returns_mean,lambda1,lambda2,lb=lb,ub=ub,tol=tol,maxiter=maxiter,approach="closed-form",verbose=False)
        if status=="failed":
            print(f" Closed Form Optimization Failed for lambda1: {lambda1} and lambda2: {lambda2}")
            continue
        print(f" lambda1: {np.round(lambda1,8)} lambda2: {np.round(lambda2,8)} error: {np.round(error,8)}")
        results[i]=(w,error)
    sorted_dict=sorted(results.items(), key=lambda x:x[1][1])
    return sorted_dict