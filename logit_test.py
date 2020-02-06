# Copyright 2020 Artem Mavrin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for the logit op."""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from scipy.special import logit as scipy_logit, expit
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from logit import logit

# Test parameters
_XS_BETWEEN_0_AND_1 = [
  0.5,
  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  [[0.1, 0.2],
   [0.3, 0.4]],
  [[[0.1, 0.2, 0.3],
    [0.3, 0.4, 0.5]],
   [[0.5, 0.6, 0.7],
    [0.7, 0.8, 0.9]]],
  np.linspace(0.0, 1.0, 101)[1:-1],
]
_DTYPES = [
  None,
  float,
  tf.dtypes.float32,
  tf.dtypes.float64,
]


def theoretical_logit_grad(x):
  """Reference implementation for the gradient of the logit function."""
  return 1 / (x * (1.0 - x))


class LogitTest(parameterized.TestCase, test.TestCase):
  @parameterized.named_parameters(
    tf_test_util.generate_combinations_with_testcase_name(
        x=_XS_BETWEEN_0_AND_1, dtype=_DTYPES))
  def test_compare_computation_with_scipy(self, x, dtype):
    if dtype is not None:
      x = tf.dtypes.cast(x, dtype=dtype)
    y = logit(x)

    self.assertAllClose(y, scipy_logit(x))
    self.assertAllClose(expit(y), x)

  @parameterized.named_parameters(
    tf_test_util.generate_combinations_with_testcase_name(
      x=_XS_BETWEEN_0_AND_1, dtype=_DTYPES))
  def test_numerical_gradient(self, x, dtype):
    if dtype is not None:
      x = tf.dtypes.cast(x, dtype=dtype)
    theoretical, numerical = tf.test.compute_gradient(logit, [x])
    self.assertAllClose(theoretical, numerical, rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
    tf_test_util.generate_combinations_with_testcase_name(
      x=_XS_BETWEEN_0_AND_1, dtype=_DTYPES))
  def test_theoretical_gradient(self, x, dtype):
    if dtype is not None:
      x = tf.dtypes.cast(x, dtype=dtype)
    x = tf.Variable(x)

    with tf.GradientTape() as tape:
      tape.watch(x)
      y = logit(x)

    dy_dx = tape.gradient(y, x)
    self.assertAllClose(dy_dx, theoretical_logit_grad(x))

  def test_extremes(self):
    self.assertTrue(tf.math.is_inf(logit(1.0)))
    self.assertTrue(tf.math.is_inf(logit(0.0)))
    self.assertTrue(tf.math.is_nan(logit(2.0)))
    self.assertTrue(tf.math.is_nan(logit(-1.0)))


if __name__ == "__main__":
  test.main()
