import tensorflow as tf
from keras_cv.layers.preprocessing.random_mix_up import RandomMixUp
import numpy as np


class RandomMixUpTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys = _random_labels()

        layer = RandomMixUp(num_classes=2, probability=1.0)
        xs, ys = layer((xs, ys))

        self.assertEqual(xs.shape, [2, 512, 512, 3])
        # one hot smoothed labels
        self.assertEqual(ys.shape, [2, 2])

    def test_label_smoothing_range(self):
        xs = tf.ones((10, 512, 512, 3))
        # randomly sample labels
        ys = _random_labels(n_classes=10, n_samples=10)

        layer = RandomMixUp(num_classes=10, probability=0.5)
        xs, ys = layer((xs, ys))

        self.assertEqual(xs.shape, [10, 512, 512, 3])
        # one hot smoothed labels
        self.assertAllInRange(ys, 0, 1)


def _random_labels(n_classes=2, n_samples=2):
    return tf.constant(
        np.eye(n_classes)[np.random.choice(n_classes, n_samples)], dtype=tf.float32
    )
