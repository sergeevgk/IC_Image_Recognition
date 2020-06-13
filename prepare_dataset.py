from glob import glob
import numpy as np
from PIL import Image
import six.moves.cPickle as pickle

num_templates = 1200


def train_dataset(templates_path):
    dataset = []
    for path in ('circle/*.png', 'square/*.png', 'rectangle/*.png'):
        for file_count, file_name in enumerate(sorted(glob(templates_path + path), key=len)):
            img = Image.open(file_name).convert('LA')  # tograyscale
            pixels = [f[0] for f in list(img.getdata())]
            dataset.append(pixels)
    labels = np.ndarray(shape=(num_templates * 3,), dtype=np.int)
    labels[0: num_templates] = 0
    labels[num_templates: num_templates * 2] = 1
    labels[num_templates * 2: num_templates * 3] = 2
    return np.array(dataset), labels


def test_dataset(data_path):
    dataset = []
    for file_count, file_name in enumerate(sorted(glob(data_path), key=len)):
        img = Image.open(file_name).convert('LA')  # tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
    return np.array(dataset)


def prepare_data():
    import os
    import joblib
    data = {'data': {}, 'description': str}
    if os.path.isfile('2d_shapes_set.pkl'):
        data = joblib.load('2d_shapes_set.pkl')
        test_set = test_dataset("data/test/*.jpg")
        data['data']['test'] = test_set
        pickle.dump(data, open('2d_shapes_set.pkl', 'wb', -1))
        return

    train_set, train_labels = train_dataset("data/templates/")
    test_set = test_dataset("data/test/*.jpg")
    data['data'] = {}
    data['data']['train'] = {}
    data['data']['train']['x'] = train_set
    data['data']['train']['y'] = train_labels
    data['data']['test'] = test_set
    data['description'] = 'train: 32x32 binary images in rgb; test : ???'

    pickle.dump(data, open('2d_shapes_set.pkl', 'wb', -1))
