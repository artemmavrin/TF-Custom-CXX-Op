"""Logit (inverse sigmoid) op definition."""

import os

import tensorflow as tf

_LOGIT_LIB = os.path.join(os.path.dirname(__file__), '_logit.so')

_logit_module = tf.load_op_library(_LOGIT_LIB)
logit = _logit_module.logit


@tf.RegisterGradient("Logit")
def _logit_grad(op, grad):
    """Gradient for the Logit op.

    Args:
      op: An `Operation`. The Logit operation being differentiated.
      grad: A `Tensor`. A gradient with respect to the output of the Logit op.

    Returns:
      A `Tensor`. The gradient with respect to the input of the Logit op.
    """
    return _logit_module.logit_grad(op.inputs[0], grad)
