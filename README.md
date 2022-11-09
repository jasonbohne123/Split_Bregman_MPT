### Penalized Mean-Variance Portfolio Optimization using Weighted Elastic Net


**Model Formulation**
- Portfolio construction via minimization of mean variance objective under L1 and L2 regularization

**Parameter Estimation**
- Utilizes biased James-Stein Estimator for mean returns
- L2 regularized covariance matrix

**Split-Bregman Algorithm**
- Reformulates original objective into two distinct problems
- Iteratively solves constrained QP and LP

**Numerical Results**
  - Out of sample performance of Split Bregman optimal portfolio outperforms equal-weighted allocation for U.S. equity data 
  
**Repo Outline**
  - Data Preprocessing
  - Split-Bregman inference and evaluation
