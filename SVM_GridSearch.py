import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def main():
    # load Data
    df = pd.read_hdf('data.h5', key='data', mode='r')

    # set data
    trX, trY = np.array(df.drop('y', axis=1)), np.array(df['y'])

    # define Support Vector Classifier
    svc = SVC(kernel='rbf')

    # set range of hyper parameters
    prm_gamma = np.power(10.0, np.arange(-5,2,1))
    prm_C = np.power(10.0, np.arange(1,5,1))

    # set prarameter grid
    param_grid = [{'C':prm_C,'gamma':prm_gamma}]

    # grid search
    gs = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)

    # fit data
    gs.fit(trX, trY)

    result = pd.DataFrame(gs.cv_results_)

    result.to_csv('result_svm.csv')

    joblib.dump(gs.best_estimator_, 'svm.pkl')

    print(gs.best_score_)
    print(gs.best_params_)
    print(result)

    # set score for graph
    scores = gs.cv_results_['mean_test_score'].reshape(len(prm_C),
                                                     len(prm_gamma))
    # plot result
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation= 'nearest', cmap=plt.cm.Greys,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(prm_gamma)), prm_gamma, rotation=45)
    plt.yticks(np.arange(len(prm_C)), prm_C)
    plt.title('F1 Score')
    plt.savefig('GridSearch.eps')
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
