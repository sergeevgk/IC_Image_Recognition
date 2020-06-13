import numpy as np
import joblib
from skimage import color
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from classifier import Classifier


class SgdClassifier(Classifier):

    def __init__(self, path_train=None, path_test=None):
        super().__init__(path_train, path_test)

    def train(self):
        self.__prepare_dataset()
        x = self.data['data']['train']['x']
        y = self.data['data']['train']['y']
        sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        sgd_clf.fit(x, y)
        self.model = sgd_clf

    def __prepare_dataset(self):
        from prepare_dataset import prepare_data
        prepare_data()
        self.data = joblib.load('2d_shapes_set.pkl')


    def test(self):
        x = self.data['data']['test']
        labels = self.model.predict(x)
        print(np.array(labels))
        return labels