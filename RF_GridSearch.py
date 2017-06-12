import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='data', mode='r')

    # set data
    trX, trY = np.array(df.drop('y', axis=1)), np.array(df['y'])

    # define Random Forest Classifier
    RF = RandomForestClassifier()

    prm_n_estimators =np.power(10, np.arange(0,6,1))
    prm_max_depth = list(range(1,49))

    # set grid
    param_grid = [{'n_estimators':prm_n_estimators, 'max_depth':prm_max_depth}]

    gs = GridSearchCV(estimator=RF, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)

    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_rf.csv')

    joblib.dump(gs.best_estimator_, 'rf.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)

    # set score for graph
    scores = gs.cv_results_['mean_test_score'].reshape(len(prm_max_depth),
                                                     len(prm_n_estimators))
    # plot result
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation= 'nearest', cmap=plt.cm.Greys,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('n_estimators')
    plt.ylabel('Max Depth')
    plt.colorbar()
    plt.xticks(np.arange(len(prm_n_estimators)), prm_n_estimators, rotation=45)
    plt.yticks(np.arange(len(prm_max_depth)), prm_max_depth)
    plt.title('F1 Score')
    plt.savefig('GridSearch_rf.eps')
    plt.show()

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

if __name__ == '__main__':
    main()
