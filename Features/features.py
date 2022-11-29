import sys
import numpy as np
import pandas as pd



def load_features(train_test_split=0.5):
    path='/home/jbohn/jupyter/personal/L1_Portfolio_Opt/data'
    equity_data = pd.read_csv(path+'/equity_data_2010_2020.csv', index_col=0)


    returns=np.log(equity_data/equity_data.shift(1))
    returns=returns.fillna(0)
    returns=returns.iloc[1:]
    cutoff=np.floor(train_test_split*len(returns.index)).astype(int)

    returns_train=returns.iloc[:cutoff]
    returns_test=returns.iloc[cutoff:]

    return returns_train, returns_test