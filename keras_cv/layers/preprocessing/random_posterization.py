# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow import keras

from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


@keras.utils.register_keras_serializable(package="keras_cv")
class RandomPosterization(VectorizedBaseImageAugmentationLayer):
    """Reduces the number of bits for each color channel.

    References:
    - [AutoAugment: Learning Augmentation Policies from Data](
        https://arxiv.org/abs/1805.09501
    )
    - [RandAugment: Practical automated data augmentation with a reduced search space](
        https://arxiv.org/abs/1909.13719
    )

    Args:
        value_range: A tuple or a list of two elements. The first value represents
            the lower bound for values in passed images, the second represents the
            upper bound. Images passed to the layer should have values within
            `value_range`.
        bits_factor: A tuple of two integers, a single integer or a
            `keras_cv.FactorSampler`. The number of bits to keep for each
            channel will be sampled from these values.
            Note that this `bits` must be between 1-8.

     Usage:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    print(images[0, 0, 0])
    # [59 62 63]
    # Note that images are Tensors with values in the range [0, 255] and uint8 dtype
    # We can pass tuple of (4, 4) to always keep the same amount of 4 bits.
    posterization = RandomPosterization(bits_factor=(4, 4), value_range=[0, 255])
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

    def __init__(self, value_range, bits_factor, seed=None, **kwargs):
        super().__init__(**kwargs)

        if not len(value_range) == 2:
            raise ValueError(
                "value_range must be a sequence of two elements. "
                f"Received: {value_range}"
            )

        self.factor = preprocessing.parse_factor(
            bits_factor,
            min_value=1.0,
            max_value=8.0,
            seed=seed,
        )

        self._value_range = value_range

    def augment_ragged_image(self, image, transformation, **kwargs):
        # TODO
        pass

    def get_random_transformation_batch(self, batch_size, **kwargs):
        bits = tf.cast(self.factor(shape=(batch_size, 1, 1, 1)), tf.uint8)
        return {"bits": bits}

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing.transform_value_range(
            images,
            original_range=self._value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )

        images = tf.cast(images, tf.uint8)
        shift = 8 - transformations["bits"]
        outputs = tf.bitwise.left_shift(
            tf.bitwise.right_shift(images, shift), shift
        )

        outputs = tf.cast(outputs, self.compute_dtype)
        return preprocessing.transform_value_range(
            outputs,
            original_range=(0, 255),
            target_range=self._value_range,
            dtype=self.compute_dtype,
        )

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def get_config(self):
        config = {
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
