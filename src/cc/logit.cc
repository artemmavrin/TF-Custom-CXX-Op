// Defines kernels for the logit (inverse sigmoid) op and gradient op.

#include <cmath>
#include <type_traits>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

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
