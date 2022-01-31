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
from keras_cv.metrics.coco import recall
import tensorflow as tf

from keras_cv.metrics.coco.base import COCOBase


class COCOMeanAveragePrecision(COCOBase):
    """COCOMeanAveragePrecision computes MaP.

    Args:
        recall_thresholds: List of floats.  Defaults to [0:.01:1].
    """

    def __init__(self, recall_thresholds=None, **kwargs):
        super().__init__(**kwargs)
        recall_thresholds = recall_thresholds or [x/100. for x in range(0, 101)]
        self.recall_thresholds = self._add_constant_weight('recall_thresholds', recall_thresholds)
        self.num_recall_thresholds = len(recall_thresholds)
        
    # [TxRxKxAxM]
    def result(self):
        present_values = self.ground_truth_boxes != 0
        n_present_categories = tf.math.reduce_sum(
            tf.cast(present_values, tf.float32), axis=-1
        )

        recalls = tf.math.divide_no_nan(
            # Broadcast ground truths to [num_thresholds, num_categories]
            self.true_positives, self.ground_truth_boxes[None, :]
        )
        precisions = tf.math.divide_no_nan(self.true_positives, self.false_positives + self.true_positives)

        tf.print("Recalls", tf.shape(recalls))
        tf.print("Precisions", tf.shape(precisions))
        precisions_at_recall  = tf.TensorArray(tf.float32, size=self.num_categories*self.num_recall_thresholds)
        for category_i in range(self.num_categories):
            inds = tf.searchsorted(recalls[:, category_i], self.recall_thresholds, side='left')
            for recall_i in range(self.num_recall_thresholds):
                if recall_i > tf.shape(inds)[0]:
                    break
                threshold = inds[recall_i]
                if threshold >= tf.shape(precisions)[0]:
                    break
                precision = precisions[threshold, category_i]
                precisions_at_recall = precisions_at_recall.write(recall_i + (category_i*self.num_recall_thresholds), precision)

        precisions_at_recall = tf.reshape(precisions_at_recall.stack(), (self.num_recall_thresholds, self.num_categories))
        # This isn't the real way to compute the metric, just a hack to test the rest of the logic.
        average_over_categories = tf.math.reduce_sum(precisions_at_recall, axis=-1)/n_present_categories
        average_over_thresholds = tf.math.reduce_mean(average_over_categories)
        return average_over_thresholds
