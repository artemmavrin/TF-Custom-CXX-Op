"""Unit tests for the logit op."""

import itertools

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from scipy.special import logit as scipy_logit, expit as scipy_sigmoid

from tf_custom_cxx_op import logit

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


# Helper functions


def _theoretical_logit_grad(x):
    """Reference implementation for the gradient of the logit function."""
    return 1 / (x * (1.0 - x))


def _named_parameters(**kwargs):
    """Generate named parameter dicts for parameterized test cases.

    E.g., _named_parameters(a=[0, 1], b=2) -> {'a': 0, 'b': 2}, {'a': 1, 'b': 2}
    """
    combinations = ([(k, v) for v in (vs if isinstance(vs, list) else [vs])]
                    for k, vs in kwargs.items())
    return list(map(dict, itertools.product(*combinations)))


def _named_parameters_with_testcase_names(**kwargs):
    """Generate named parameter dicts with test names for parameterized test
    cases.
    """
    named_parameters = _named_parameters(**kwargs)
    index_padding = len(str(len(named_parameters)))
    for index, kwargs in enumerate(named_parameters):
        name = f'_{index:0{index_padding}}'
        for key, value in sorted(kwargs.items(), key=lambda pair: pair[0]):
            key = ''.join(filter(str.isalnum, str(key)))
            value = ''.join(filter(str.isalnum, str(value)))
            name += f'_{key}_{value}'
        kwargs['testcase_name'] = name
    return named_parameters


class LogitTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.named_parameters(_named_parameters_with_testcase_names(
        x=_XS_BETWEEN_0_AND_1,
        dtype=_DTYPES),
    )
    def test_compare_computation_with_scipy(self, x, dtype):
        if dtype is not None:
            x = tf.dtypes.cast(x, dtype=dtype)
        y = logit(x)

        self.assertAllClose(y, scipy_logit(x))
        self.assertAllClose(scipy_sigmoid(y), x)

    @parameterized.named_parameters(_named_parameters_with_testcase_names(
        x=_XS_BETWEEN_0_AND_1,
        dtype=_DTYPES),
    )
    def test_numerical_gradient(self, x, dtype):
        if dtype is not None:
            x = tf.dtypes.cast(x, dtype=dtype)
        theoretical, numerical = tf.test.compute_gradient(logit, [x])
        self.assertAllClose(theoretical, numerical, rtol=1e-2, atol=1e-2)

    @parameterized.named_parameters(_named_parameters_with_testcase_names(
        x=_XS_BETWEEN_0_AND_1,
        dtype=_DTYPES),
    )
    def test_theoretical_gradient(self, x, dtype):
        if dtype is not None:
            x = tf.dtypes.cast(x, dtype=dtype)
        x = tf.Variable(x)

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = logit(x)

        dy_dx = tape.gradient(y, x)
        self.assertAllClose(dy_dx, _theoretical_logit_grad(x))

    def test_extremes(self):
        self.assertTrue(tf.math.is_inf(logit(1.0)))
        self.assertTrue(tf.math.is_inf(logit(0.0)))
        self.assertTrue(tf.math.is_nan(logit(2.0)))
        self.assertTrue(tf.math.is_nan(logit(-1.0)))


if __name__ == "__main__":
    tf.test.main()
