### Penalized Mean-Variance Portfolio Optimization using Weighted Elastic Net


**Portfolio Optimization**
- Portfolio construction via minimization of mean variance objectivefor sequence of target mean return across efficient frontier
- Includes support for L1 and L2 Regularization via Split Bregman Algorithm

**Parameter Estimation**
- Utilizes biased James-Stein Estimator for mean returns
- L2 regularized covariance matrix

**Split-Bregman Algorithm**
- Reformulates original objective into two distinct problems
- Iteratively solves constrained QP and LP
- Performs a grid search for optimal calibration of regularization parameters 

**Out of Sample Numerical Results**
- Minimum Variance Objective
- Mean Variance Objective 
- Biased Mean Variance Objective
- Unbiased Mean Variance Objective with Elastic Net Penalty
- Biased Mean Variance Objective with Elastic Net Penalty

All results evaluated on out of sample U.S. equity data 
  

