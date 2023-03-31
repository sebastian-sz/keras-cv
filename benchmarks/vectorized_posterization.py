import time

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from matplotlib import pyplot as plt

from keras_cv.layers import BaseImageAugmentationLayer
from keras_cv.layers import RandomPosterization
from keras_cv.utils import transform_value_range


class OldPosterization(BaseImageAugmentationLayer):
    """Reduces the number of bits for each color channel.

    References:
    - [AutoAugment: Learning Augmentation Policies from Data](
        https://arxiv.org/abs/1805.09501
    )
    - [RandAugment: Practical automated data augmentation with a reduced search space](
        https://arxiv.org/abs/1909.13719
    )

    Args:
        value_range: a tuple or a list of two elements. The first value represents
            the lower bound for values in passed images, the second represents the
            upper bound. Images passed to the layer should have values within
            `value_range`. Defaults to `(0, 255)`.
        bits: integer. The number of bits to keep for each channel. Must be a value
            between 1-8.

     Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    print(images[0, 0, 0])
    # [59 62 63]
    # Note that images are Tensors with values in the range [0, 255] and uint8 dtype
    posterization = Posterization(bits=4, value_range=[0, 255])
    images = posterization(images)
    print(images[0, 0, 0])
    # [48., 48., 48.]
    # NOTE: the layer will output values in tf.float32, regardless of input dtype.
    ```

     Call arguments:
        inputs: input tensor in two possible formats:
            1. single 3D (HWC) image or 4D (NHWC) batch of images.
            2. A dict of tensors where the images are under `"images"` key.
    """

    def __init__(self, value_range, bits, **kwargs):
        super().__init__(**kwargs)

        if not len(value_range) == 2:
            raise ValueError(
                "value_range must be a sequence of two elements. "
                f"Received: {value_range}"
            )

        if not (0 < bits < 9):
            raise ValueError(
                f"Bits value must be between 1-8. Received bits: {bits}."
            )

        self._shift = 8 - bits
        self._value_range = value_range

    def augment_image(self, image, **kwargs):
        image = transform_value_range(
            images=image,
            original_range=self._value_range,
            target_range=[0, 255],
        )
        image = tf.cast(image, tf.uint8)

        image = self._posterize(image)

        image = tf.cast(image, self.compute_dtype)
        return transform_value_range(
            images=image,
            original_range=[0, 255],
            target_range=self._value_range,
            dtype=self.compute_dtype,
        )

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    def _batch_augment(self, inputs):
        # Skip the use of vectorized_map or map_fn as the implementation is already
        # vectorized
        return self._augment(inputs)

    def _posterize(self, image):
        return tf.bitwise.left_shift(
            tf.bitwise.right_shift(image, self._shift), self._shift
        )

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def get_config(self):
        config = {"bits": 8 - self._shift, "value_range": self._value_range}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PosterizationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(list(range(1, 9)))
    def test_consistency_with_old_impl(self, bits):
        image_shape = (3, 32, 32, 3)
        image = tf.random.uniform(shape=image_shape, minval=0, maxval=255)

        layer = RandomPosterization(
            value_range=(0, 255), bits_factor=(bits, bits)
        )
        old_layer = OldPosterization(value_range=(0, 255), bits=bits)

        output = layer(image)
        old_output = old_layer(image)

        self.assertNotAllClose(image, output)
        self.assertAllClose(old_output, output)


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 5000, 10000]
    results = {}
    aug_candidates = [RandomPosterization, OldPosterization]
    old_aug_args = {"value_range": (0, 255), "bits": 4}
    new_aug_args = {"value_range": (0, 255), "bits_factor": (4, 4)}

    for aug in aug_candidates:
        # Eager Mode
        c = aug.__name__

        if aug == RandomPosterization:
            aug_args = new_aug_args
        else:
            aug_args = old_aug_args

        layer = aug(**aug_args)
        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            layer(x_train[:n_images])

            t0 = time.time()
            r1 = layer(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")
        results[c] = runtimes

        # Graph Mode
        c = aug.__name__ + " Graph Mode"
        layer = aug(**aug_args)

        @tf.function()
        def apply_aug(inputs):
            return layer(inputs)

        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            apply_aug(x_train[:n_images])

            t0 = time.time()
            r1 = apply_aug(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")
        results[c] = runtimes

        # XLA Mode
        c = aug.__name__ + " XLA Mode"
        layer = aug(**aug_args)

        @tf.function(jit_compile=True)
        def apply_aug(inputs):
            return layer(inputs)

        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            apply_aug(x_train[:n_images])

            t0 = time.time()
            r1 = apply_aug(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")
        results[c] = runtimes

    plt.figure()
    for key in results:
        plt.plot(num_images, results[key], label=key)
        plt.xlabel("Number images")

    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.savefig("comparison.png")

    # So we can actually see more relevant margins
    del results[aug_candidates[1].__name__]
    plt.figure()
    for key in results:
        plt.plot(num_images, results[key], label=key)
        plt.xlabel("Number images")

    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.savefig("comparison_no_old_eager.png")

    # Run unit tests
    tf.test.main()
