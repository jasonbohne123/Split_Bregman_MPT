### Mean-Variance Portfolio Optimization using  Elastic Net Penalty

**Data Preprocessing and Feature Generation**
- U.S. Equity data for underlyings in S&P500 between 2010 and 2021
- Assets remvoed when missing more than 1% of observations
- Asset universe consists of 439 potential equities

**Parameter Estimation**
- Utilizes biased James-Stein Estimator for mean returns
- L2 regularized covariance matrix

**Portfolio Optimization**
- Portfolio construction via minimization of mean variance objective across efficient frontier
- Includes support for L1 and L2 Regularization via Split Bregman Algorithm

**Split-Bregman Algorithm**
- Reformulates original objective into two distinct problems
- Iteratively solves constrained QP and LP either in closed-form or numerically
- Performs a grid search for optimal calibration of regularization parameters 

**Numerical Results**

Approaches:
- Minimum Variance Objective
- Mean Variance Objective 
- Biased Mean Variance Objective
- Unbiased Mean Variance Objective with Elastic Net Penalty
- Biased Mean Variance Objective with Elastic Net Penalty

All results evaluated on out of sample U.S. equity data 
  

