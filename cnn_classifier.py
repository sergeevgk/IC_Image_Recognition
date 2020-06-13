import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from classifier import Classifier

LABELS = ['circle', 'rectangle', 'square']


class CnnClassifier(Classifier):

    def __init__(self, path_train=None, path_test=None):
        super().__init__(path_train, path_test)

    def train(self):
        model, trained = make_or_restore_model('data/cnn_checkpoints')
        if trained:
            self.model = model
            self.data = {}
            self.data['test'] = create_test_ds(path='data/test')
            return
        self.__prepare_dataset()
        ds, image_count = self.data['train']
        BATCH_SIZE = 32
        # Установка размера буфера перемешивания, равного набору данных, гарантирует
        # полное перемешивание данных.
        ds = ds.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='data/cnn_checkpoints/ckpt-loss={loss:.2f}',
                save_freq=100)
        ]
        steps_per_epoch = tf.math.ceil(image_count / BATCH_SIZE).numpy()
        model.fit(ds, epochs=8, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        self.model = model
        pass

    def __prepare_dataset(self):
        self.data = {}
        self.data['train'] = create_ds(path='data/templates')
        self.data['test'] = create_test_ds(path='data/test')
        pass

    def test(self):
        test_data, num = self.data['test']
        labels = self.model.predict(test_data)
        k = 0
        for l in labels:
            index_max = np.argmax(l)
            print(str(k) + " : " + LABELS[index_max])
            k = k + 1
        return labels


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    # image = tf.image.resize(image, [192, 192])
    # image /= 255.0  # normalize to [0,1] range

    return image


def create_ds(path):
    import pathlib
    import random
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    print(image_ds)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    print(image_label_ds)

    return image_label_ds, image_count


def create_test_ds(path):
    import pathlib
    import cv2
    image_count = len(path)
    path = pathlib.Path(path)
    all_image_paths = list(path.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    arr = []
    for p in all_image_paths:
        arr.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    arr = np.array(arr)
    return arr, image_count


def make_model():
    # Create a new linear regression model.
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32)),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def make_or_restore_model(checkpoint_dir):
    import os

    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint), True
    print('Creating a new model')
    return make_model(), False
