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
"""Logit (inverse sigmoid) op definition."""

import tensorflow as tf

_logit_module = tf.load_op_library('_logit.so')
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
