/* Copyright 2020 Artem Mavrin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Registers the logit (inverse sigmoid) op and gradient op.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

REGISTER_OP("Logit")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double}")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape)
    .Doc("Inverse of the sigmoid function, `logit(x) = log(x / (1 - x))`.");

REGISTER_OP("LogitGrad")
    .Input("x: T")
    .Input("dz_dy: T")
    .Output("dz_dx: T")
    .Attr("T: {float, double}")
    .SetShapeFn(tensorflow::shape_inference::MergeBothInputsShapeFn);
