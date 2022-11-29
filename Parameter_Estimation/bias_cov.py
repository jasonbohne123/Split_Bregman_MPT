import numpy as np

def reg_cov(cov,penalty=0.5):
    return (1-penalty)*cov+penalty*np.identity(cov.shape[0])