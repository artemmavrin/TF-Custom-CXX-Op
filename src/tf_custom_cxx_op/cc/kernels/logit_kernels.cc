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

// Defines kernels for the logit (inverse sigmoid) op and gradient op.

#include <cmath>
#include <type_traits>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace functor {
    template <typename T>
    class Logit {
    public:
        void operator()(typename tensorflow::TTypes<T>::ConstTensor x,
                        typename tensorflow::TTypes<T>::Tensor y) {
            T p;
            const auto n = x.size();
            typename std::remove_const<decltype(n)>::type i;
            for (i = 0; i < n; ++i) {
                p = x(i);
                y(i) = log(p / (1 - p));
            }
        }
    };

    template <typename T>
    class LogitGrad {
    public:
        void operator()(typename tensorflow::TTypes<T>::ConstFlat x,
                        typename tensorflow::TTypes<T>::ConstFlat dz_dy,
                        typename tensorflow::TTypes<T>::Flat dz_dx) {
            T p;
            const auto n = x.size();
            typename std::remove_const<decltype(n)>::type i;
            for (i = 0; i < n; ++i) {
                p = x(i);
                dz_dx(i) = dz_dy(i) / (p * (1 - p));
            }
        }
    };
}  // namespace functor

template <typename T>
class LogitOp : public tensorflow::UnaryElementWiseOp<T, LogitOp<T>> {
public:
    using tensorflow::UnaryElementWiseOp<T, LogitOp<T>>::UnaryElementWiseOp;

    void Operate(tensorflow::OpKernelContext* context,
               const tensorflow::Tensor& x,
               tensorflow::Tensor* y) {
        functor::Logit<T> functor;
        functor(x.flat<T>(), y->flat<T>());
    }
};

template <typename T>
class LogitGradOp : public tensorflow::BinaryElementWiseOp<T, LogitGradOp<T>> {
public:
    using tensorflow::BinaryElementWiseOp<T, LogitGradOp<T>>::BinaryElementWiseOp;

    template<int NDIMS>
    void Operate(tensorflow::OpKernelContext* context,
                 const tensorflow::Tensor& x,
                 const tensorflow::Tensor& dz_dy,
                 tensorflow::Tensor* dz_dx) {
        functor::LogitGrad<T> functor;
        functor(x.flat<T>(), dz_dy.flat<T>(), dz_dx->flat<T>());
    }
};

#define REGISTER_KERNEL(name, Op, T)        \
    REGISTER_KERNEL_BUILDER(                \
        Name(name)                          \
            .Device(tensorflow::DEVICE_CPU) \
            .TypeConstraint<T>("T"),        \
        Op<T>                               \
    )

REGISTER_KERNEL("Logit", LogitOp, float);
REGISTER_KERNEL("Logit", LogitOp, double);
REGISTER_KERNEL("LogitGrad", LogitGradOp, float);
REGISTER_KERNEL("LogitGrad", LogitGradOp, double);

#undef REGISTER_KERNEL
