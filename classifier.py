import numpy as np


class Classifier:
    def __init__(self, path_train=None, path_test=None):
        self.path_train = path_train
        self.path_test = path_test
        self.model = None
        self.data=None

    def test(self, model, data):
        pass

    def train(self):
        pass

    def get_model(self):
        return self.model

    def prepare_datasets(self):
        pass
