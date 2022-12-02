import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import csc_matrix
from qpsolvers import solve_qp


def qp(cov,mean,w,d,b,lb,ub,lambda2,approach,tol=10e-6,maxiter=25,verbose=True):
    """ Inner optimization problem using unconstrained optimization 
        Approach "numerical" solves numerically via unconstrained minimization
        Approach "closed-form" solves analytically via cvxopt within qpsolvers 
        Support for standard mean variance optimization (approach=2; pen=False)
    """
    
 
    x0=w
    status=None

    if approach=="numerical":
        
        # define objective function
        def objective_val(w,cov,mean,d,b,lambda2):
            obj=np.dot(np.dot(w,cov+lambda2*np.diag(d-b)),w)-np.dot(w,mean)
            return obj
        
        # define bounds and constraints
        bounds=Bounds(lb,ub)
        lc=LinearConstraint(np.ones(len(mean)),1,1)

        # solve unconstrained optimization problem
        try:
            res=minimize(objective_val,x0=x0,args=(cov,mean,d,b,lambda2),method='SLSQP',bounds=bounds,constraints=lc,options={'maxiter':maxiter},tol=tol)
        except Exception as e:
            print(e)
            status="failed"
            return w,status
        # check if optimization was successful
        w=res.x
        status=res.status
        
    if approach=="closed-form":
        w_old=w
        
        # define objective function
        quad=csc_matrix((cov+lambda2*np.diag(d-b)),shape=(len(mean),len(mean)))
        linear=-1*mean
        
        # constrain solution vector to be unit norm and non-negative
        A=np.ones(len(mean))
        constraint=np.array([1.0])

        # set box constraints
        lb_vec=lb*np.ones(len(mean))
        ub_vec=ub*np.ones(len(mean))

        #solve qp problem
        try:
            w = solve_qp(quad,linear,A=A,b=constraint,lb=lb_vec,ub=ub_vec,initvals=x0, solver="cvxopt")
        except Exception as e:
            print(e)
            status="failed"
            return w_old,status
        status="solved"

        if w is None:
            print("QP Optimization fails, Trying again")
            status="failed"
            return w_old,status
    return w,status

def shrinkage(w,b,lambda1,beta):
    """ Applies one-sided soft-thresholding operator that is equivalent to L1 penalization 
    """
    
    if lambda1==0:
        return np.zeros(len(w))
    x=beta*w+b
    obj=np.fmax([np.abs(x)-(1/lambda1)],[np.zeros(len(w))])  
    signed_obj=np.sign(x)*obj

    return signed_obj[0]

def split_bregman(cov,mean,lambda1,lambda2,lb,ub,beta=None,tol=1e-10,maxiter=100,approach=2,verbose=True):
    """ Split Bregman Optimization Routine
    """

    total_cost=1
    n=len(mean)
    if beta is None:
        beta=np.ones(n)
    
    b=np.zeros(n)
    w=np.random.rand(n)
    w=w/np.linalg.norm(w,ord=1)
    d=np.zeros(n)
    
    i=0
    while (total_cost)>tol:
        if i>maxiter:
            if verbose:
                print("MaxIter Achieved")
            break
        
        # solve qp problem either closed form or numerically
        w,status=qp(cov,mean,w,d,b,lb,ub,lambda2,approach,tol,maxiter,verbose)
        
        if status=="failed":
            break
            
 
        # solve for sparse vector near previous optimal solution
        d=shrinkage(w,b,lambda1,beta)
        b=b+beta*w -d 
        i+=1

        # compute total cost
        total_cost=np.dot(np.dot(w,cov),w)-np.dot(w,mean)+(lambda1*np.sum(np.abs(w))+lambda2*np.sum((w)**2))/2
        if verbose:
            print(f"Total Cost: {total_cost}")
        
        w_old=w
    return w, total_cost,status