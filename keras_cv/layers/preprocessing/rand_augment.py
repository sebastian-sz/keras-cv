# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NAME_TO_FUNC = {
    "AutoContrast": autocontrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": wrapped_rotate,
    "Posterize": posterize,
    "Solarize": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "Contrast": contrast,
    "Brightness": brightness,
    "Sharpness": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x,
    "TranslateY": translate_y,
    "Cutout": cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset(
    {
        "Rotate",
        "TranslateX",
        "ShearX",
        "ShearY",
        "TranslateY",
        "Cutout",
    }
)


def level_to_arg(cutout_const: float, translate_const: float):
    """Creates a dict mapping image operation names to their arguments."""

    no_arg = lambda level: ()
    posterize_arg = lambda level: _mult_to_arg(level, 4)
    solarize_arg = lambda level: _mult_to_arg(level, 256)
    solarize_add_arg = lambda level: _mult_to_arg(level, 110)
    cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
    translate_arg = lambda level: _translate_level_to_arg(level, translate_const)

    args = {
        "AutoContrast": no_arg,
        "Equalize": no_arg,
        "Invert": no_arg,
        "Rotate": _rotate_level_to_arg,
        "Posterize": posterize_arg,
        "Solarize": solarize_arg,
        "SolarizeAdd": solarize_add_arg,
        "Color": _enhance_level_to_arg,
        "Contrast": _enhance_level_to_arg,
        "Brightness": _enhance_level_to_arg,
        "Sharpness": _enhance_level_to_arg,
        "ShearX": _shear_level_to_arg,
        "ShearY": _shear_level_to_arg,
        "Cutout": cutout_arg,
        "TranslateX": translate_arg,
        "TranslateY": translate_arg,
    }
    return args


def _parse_policy_info(
    name,
    prob,
    level,
    replace_value,
    cutout_const,
    translate_const,
    level_std,
):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]

    if level_std > 0:
        level += tf.random.normal([], dtype=tf.float32)
        level = tf.clip_by_value(level, 0.0, _MAX_LEVEL)

    args = level_to_arg(cutout_const, translate_const)[name](level)

    if name in REPLACE_FUNCS:
        # Add in replace arg if it is required for the function that is called.
        args = tuple(list(args) + [replace_value])

    return func, prob, args


class RandAugment(layers.Layer):
    """Applies RandAugment to images.  RandAugment is a
    common preprocessing component in state of the art
    image classification pipelines.  The RandAugment
    policy consists of constrast adjstments, croppings,
    translations, color shifts, brightness shifts,
    rotations and more.  For the average practicioner,
    RandAugment can be considered a "one stop shop"
    for image data augmentation.

    References:
    - https://arxiv.org/abs/1909.13719
    """
    def __init__(self,
        num_layers,
        magnitude,
        cutout_const = 40.0,
        translate_const = 100.0,
        magnitude_std = 0.0
    ):
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.cutout_const = float(cutout_const)
        self.translate_const = float(translate_const)
        self.prob_to_apply = float(prob_to_apply) if prob_to_apply is not None else None
        self.available_ops = [
            "AutoContrast",
            "Equalize",
            "Invert",
            "Rotate",
            "Posterize",
            "Solarize",
            "Color",
            "Contrast",
            "Brightness",
            "Sharpness",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
            "Cutout",
            "SolarizeAdd",
        ]
        self.magnitude_std = magnitude_std
        if exclude_ops:
            self.available_ops = [
                op for op in self.available_ops if op not in exclude_ops
            ]

    def call(self, images):
       input_image_type = image.dtype

        if input_image_type != tf.uint8:
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, dtype=tf.uint8)

        replace_value = [128] * 3
        min_prob, max_prob = 0.2, 0.8

        aug_image = image

        for _ in range(self.num_layers):
            op_to_select = tf.random.uniform(
                [], maxval=len(self.available_ops) + 1, dtype=tf.int32
            )

            branch_fns = []
            for (i, op_name) in enumerate(self.available_ops):
                prob = tf.random.uniform(
                    [], minval=min_prob, maxval=max_prob, dtype=tf.float32
                )
                func, _, args = _parse_policy_info(
                    op_name,
                    prob,
                    self.magnitude,
                    replace_value,
                    self.cutout_const,
                    self.translate_const,
                    self.magnitude_std,
                )
                branch_fns.append(
                    (
                        i,
                        # pylint:disable=g-long-lambda
                        lambda selected_func=func, selected_args=args: selected_func(
                            image, *selected_args
                        ),
                    )
                )
                # pylint:enable=g-long-lambda

            aug_image = tf.switch_case(
                branch_index=op_to_select,
                branch_fns=branch_fns,
                default=lambda: tf.identity(image),
            )

            if self.prob_to_apply is not None:
                aug_image = tf.cond(
                    tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
                    lambda: tf.identity(aug_image),
                    lambda: tf.identity(image),
                )
            image = aug_image

        image = tf.cast(image, dtype=input_image_type)
        return image
