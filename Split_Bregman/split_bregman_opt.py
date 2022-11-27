import numpy as np
import scipy.optimize
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix, csc_array



def qp(cov,mean,w,d,b,beta,lambda2,tol,maxiter,approach,pen=True):
    """ Inner optimization problem using unconstrained optimization 
        Approach 1 solves numerically via unconstrained minimization
        Approach 2 solves analytically via cvxopt within qpsolvers 
        Support for standard mean variance optimization (approach=2; pen=False)
    """
    
    x0=w # initial guess is previous result
    
    
    if approach==1:
        def objective_val(w,cov,mean,d,b,beta,lambda2):
            obj=np.dot(np.dot(w,cov),w)-np.dot(w,mean)

            if pen:
                l2_pen=np.sum((d-beta*w-b)**2)
            else:
                l2_pen=0

            return obj+lambda2*l2_pen/2

        res=scipy.optimize.minimize(objective_val,x0=w,args=(returns_cov,returns_mean,d,b,beta,lambda2),method='CG',options={'maxiter':25},tol=1e-12)
        w=res.x
        
    if approach==2:
        
        w_old=w
        if pen:
            quad=csc_matrix(2*(cov+2*lambda2*np.identity(len(mean))))
            linear=-1*mean+lambda2*(b-d)
        else:
            quad=csc_matrix(2*(cov))
            linear=-1*mean
        
        w = solve_qp(quad,linear,initvals=x0, solver="cvxopt")

        if w is None:
            return w_old
        
    return w

def shrinkage(w,d,b,lambda1,beta):
    """ Applies soft-thresholding operator that is equivalent to L1 penalization 
    """
    
    x=beta*w+b
 
    obj=np.fmax([np.abs(x)-(1/lambda1)],[np.zeros(len(w))])  # elementwise max
    signed_obj=np.sign(x)*obj

    return signed_obj[0]

def split_bregman(cov,mean,lambda1,lambda2,beta=None,tol=1e-10,maxiter=100,approach=2):
    """ Split Bregman Optimization Routine
    """

    error=1
    n=len(mean)
    if beta is None:
        beta=np.ones(n)
    
    b=np.zeros(n)
    w=np.zeros(n)
    d=np.zeros(n)
    
    
    i=0
    while error>tol:
        if i>maxiter:
            print("MaxIter Achieved")
            break
            
        
        if i%25==0 and i>0:
            print(f"{i}-th iteration with error of {error}")
    
        w=qp(cov,mean,w,d,b,beta,lambda2,tol,maxiter,approach)
        w=w/np.linalg.norm(w,ord=1)
        
        # sparse soln near w
        d=shrinkage(w,d,b,lambda1,beta)
        if sum(abs(d))!=0:
            d=d/np.linalg.norm(d,ord=1)
        
        # b difference between l2 and l1
        b=b+beta*w -d 
        if sum(abs(b))!=0:
            b=b/np.linalg.norm(b,ord=1)
        i+=1
        # error in terms of total soln
        error=abs(np.dot(np.dot(w,cov),w)-np.dot(w.T,mean)+np.sum(abs(w))+np.sum(w**2))
    
    print(f"Terminated in {i} iterations of error {error} for lambda1={lambda1} and lambda2={lambda2}")
    w=np.round(w,8)
    if sum(abs(w))==0:
        return w,error
    unit_w=w/np.linalg.norm(w,ord=1)
    return unit_w, error