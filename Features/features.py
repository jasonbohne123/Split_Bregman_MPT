import sys
import numpy as np
import pandas as pd



def load_features(train_test_split=0.5,frac=0.01):
    path='/home/jbohn/jupyter/personal/L1_Portfolio_Opt/data'
    equity_data = pd.read_csv(path+'/equity_data_2010_2020.csv', index_col=0)

    # optimizer is sensitive to scaling of features
    returns=100*np.log(equity_data/equity_data.shift(1))

    # drop assets with greater than frac missing values
    frac_none={}
    for col in returns.columns:
        frac_none[col]=True if returns[col].isna().sum()/len(returns[col])>frac else False
    
    for key, value in frac_none.items():
        if value:
            returns.drop(key,axis=1,inplace=True)
    
    returns.fillna(0,inplace=True)
    returns=returns.iloc[1:]
    cutoff=np.floor(train_test_split*len(returns.index)).astype(int)

    returns_train=returns.iloc[:cutoff]
    returns_test=returns.iloc[cutoff:]

    return returns_train, returns_test