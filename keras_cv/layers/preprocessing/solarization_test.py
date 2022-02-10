# Copyright 2022 The KerasCV Authors
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

from keras_cv.layers.preprocessing.solarization import Solarization


class SolarizationTest(tf.test.TestCase):
    def test_raises_error_on_invalid_min_max_values(self):
        with self.assertRaises(AssertionError):
            Solarization(min_value=255, max_value=0)  # min must be smaller than max

    def test_range_0_to_255(self):
        solarization = Solarization(min_value=0, max_value=255)

        test_parameters = [
            {"input_value": 0, "expected_output_value": 255},
            {"input_value": 64, "expected_output_value": 191},
            {"input_value": 127, "expected_output_value": 128},
            {"input_value": 191, "expected_output_value": 64},
            {"input_value": 255, "expected_output_value": 0},
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.uint8,
            )

    def test_range_0_to_1(self):
        solarization = Solarization(min_value=0.0, max_value=1.0)

        test_parameters = [
            {"input_value": 1.0, "expected_output_value": 0.0},
            {"input_value": 0.75, "expected_output_value": 0.25},
            {"input_value": 0.5, "expected_output_value": 0.5},
            {"input_value": 0.25, "expected_output_value": 0.75},
            {"input_value": 0.0, "expected_output_value": 1.0},
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.float32,
            )

    def test_range_minus_1_to_plus_1(self):
        solarization = Solarization(min_value=-1.0, max_value=1.0)

        test_parameters = [
            {"input_value": -1.0, "expected_output_value": 1.0},
            {"input_value": -0.5, "expected_output_value": 0.5},
            {"input_value": 0.0, "expected_output_value": 0.0},
            {"input_value": 0.5, "expected_output_value": -0.5},
            {"input_value": 1.0, "expected_output_value": -1.0},
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.float32,
            )

    def test_custom_range_minus_8_to_plus_16(self):
        solarization = Solarization(min_value=-8, max_value=16)

        test_parameters = [
            {"input_value": -8, "expected_output_value": 16},
            {"input_value": -2, "expected_output_value": 10},
            {"input_value": 4, "expected_output_value": 4},
            {"input_value": 10, "expected_output_value": -2},
            {"input_value": 16, "expected_output_value": -8},
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.int32,
            )

    @staticmethod
    def _test_input_output(layer, input_value, expected_value, dtype):
        dummy_input = tf.ones(shape=(2, 224, 224, 3), dtype=dtype) * input_value
        expected_output = tf.ones(shape=(2, 224, 224, 3), dtype=dtype) * expected_value

        output = layer(dummy_input)

        tf.debugging.assert_equal(output, expected_output)

    def test_only_values_above_threshold_are_solarized_if_threshold_specified(self):
        solarization = Solarization(min_value=0, max_value=255, threshold=128)

        test_parameters = [
            {"input_value": 0, "expected_output_value": 0},
            {"input_value": 64, "expected_output_value": 64},
            {"input_value": 127, "expected_output_value": 127},
            {"input_value": 191, "expected_output_value": 64},
            {"input_value": 255, "expected_output_value": 0},
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.uint8,
            )

    def test_input_values_outside_of_specified_range_are_clipped(self):
        solarization = Solarization(min_value=0, max_value=255)

        test_parameters = [
            {"input_value": -100, "expected_output_value": 255},  # Clipped to 0
            {"input_value": -1, "expected_output_value": 255},  # Clipped to 0
            {"input_value": 256, "expected_output_value": 0},  # Clipped to 255
            {"input_value": 300, "expected_output_value": 0},  # Clipped to 255
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.int32,
            )
