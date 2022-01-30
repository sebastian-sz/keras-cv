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

from keras_cv.metrics.coco.base import COCOBase


class COCOMeanAveragePrecision(COCOBase):
    """COCOMeanAveragePrecision computes MaP.

    Args:
        recall_thresholds: List of floats.  Defaults to [0:.01:1].
    """

    def __init__(self, recall_thresholds=None, **kwargs):
        super.__init__(**kwargs)

    # [TxRxKxAxM]
    def result(self):
        """
        if nd:
            recall[t, k, a, m] = rc[-1]
        else:
            recall[t, k, a, m] = 0

        # numpy is slow without cython optimization for accessing elements
        # use python array gets significant speed improvement
        pr = pr.tolist()
        q = q.tolist()

        for i in range(nd - 1, 0, -1):
            if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

        inds = np.searchsorted(rc, p.recThrs, side='left')
        try:
            for ri, pi in enumerate(inds):
            q[ri] = pr[pi]
        except:
            pass
        precision[t, :, k, a, m] = np.array(q)
        """
        return 0.0
