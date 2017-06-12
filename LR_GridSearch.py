import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='data', mode='r')

    # set data
    trX, trY = np.array(df.drop('y', axis=1)), np.array(df['y'])

    # set Logistic Regression
    lr = LogisticRegression()

    prm_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 100.0, 1000.0] # set parameter range for C
    penalty_range = ['l1', 'l2'] # set penalty range
    solver_range = ['lbfgs'] # set solver range
    iter_range = [1000] # set range for max_iter

    # set grid
    param_grid = [{'C':prm_range, 'penalty':['l1'],'solver':['liblinear'],'max_iter':iter_range},\
    {'C':prm_range, 'penalty':['l2'],'solver':solver_range, 'max_iter':iter_range}]

    gs = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_lr.csv')

    joblib.dump(gs.best_estimator_, 'lr.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)


if __name__ == '__main__':
    main()
