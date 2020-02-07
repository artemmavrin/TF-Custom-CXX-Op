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

// Defines the logit (inverse sigmoid) op and gradient op.

#include <cmath>
#include <type_traits>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

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

template<typename T>
class LogitOp : public tensorflow::UnaryElementWiseOp<T, LogitOp<T>> {
public:
  using tensorflow::UnaryElementWiseOp<T, LogitOp<T>>::UnaryElementWiseOp;

  void Operate(tensorflow::OpKernelContext* context,
               const tensorflow::Tensor& x,
               tensorflow::Tensor* y) {
    // Flat views into the input and output tensors
    auto x_flat = x.flat<T>();
    auto y_flat = y->flat<T>();

    // Compute logits from probabilities
    const auto n = x_flat.size();
    typename std::remove_const<decltype(n)>::type i;
    T p;
    for (i = 0; i < n; ++i) {
      p = x_flat(i);
      y_flat(i) = log(p / (1.0 - p));
    }
  }
};

template<typename T>
class LogitGradOp : public tensorflow::BinaryElementWiseOp<T, LogitGradOp<T>> {
public:
  using tensorflow::BinaryElementWiseOp<T, LogitGradOp<T>>::BinaryElementWiseOp;

  template<int NDIMS>
  void Operate(tensorflow::OpKernelContext* context,
               const tensorflow::Tensor& x,
               const tensorflow::Tensor& dz_dy,
               tensorflow::Tensor* dz_dx) {
    // Flat views into the input and output tensors
    auto x_flat = x.flat<T>();
    auto dz_dy_flat = dz_dy.flat<T>();
    auto dz_dx_flat = dz_dx->flat<T>();

    // Compute back propagated gradients
    const auto n = x_flat.size();
    typename std::remove_const<decltype(n)>::type i;
    T p;
    for (i = 0; i < n; ++i) {
      p = x_flat(i);
      dz_dx_flat(i) = dz_dy_flat(i) / (p * (1.0 - p));
    }
  }
};

#define REGISTER_KERNEL(name, Op, T)  \
  REGISTER_KERNEL_BUILDER(            \
    Name(name)                        \
      .Device(tensorflow::DEVICE_CPU) \
      .TypeConstraint<T>("T"),        \
    Op<T>                             \
  )

REGISTER_KERNEL("Logit", LogitOp, float);
REGISTER_KERNEL("Logit", LogitOp, double);
REGISTER_KERNEL("LogitGrad", LogitGradOp, float);
REGISTER_KERNEL("LogitGrad", LogitGradOp, double);

#undef REGISTER_KERNEL
