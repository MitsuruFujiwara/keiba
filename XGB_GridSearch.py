import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='data', mode='r')

    # set data
    trX, trY = np.array(df.drop('y', axis=1)), np.array(df['y'])

    # define Xgboost Classifier
    XGB = XGBClassifier()

    prm_max_depth =list(range(1,10))
    prm_lr = np.power(10.0, np.arange(-10,0,1))

    param_grid = [{'learning_rate':prm_lr, 'max_depth':prm_max_depth, }]

    gs = GridSearchCV(estimator=XGB, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_xgb.csv')

    joblib.dump(gs.best_estimator_, 'xgb.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)

if __name__ == '__main__':
    main()
