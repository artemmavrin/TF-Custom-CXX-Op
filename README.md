# TensorFlow Custom C++ Op

[![Build](https://github.com/artemmavrin/TF-Custom-CXX-Op/workflows/Build/badge.svg "Build")](https://github.com/artemmavrin/TF-Custom-CXX-Op/actions?query=workflow%3ABuild)
[![Last Commit](https://img.shields.io/github/last-commit/artemmavrin/TF-Custom-CXX-Op/master "Last Commit")](https://github.com/artemmavrin/TF-Custom-CXX-Op)
[![License](https://img.shields.io/github/license/artemmavrin/TF-Custom-CXX-Op "License")](https://github.com/artemmavrin/TF-Custom-CXX-Op/blob/master/LICENSE)

**This Repo Is Under Construction**

This is an example of how to create a custom TensorFlow op in C++ and use it in Python.

The op implemented here is the [logit function](https://en.wikipedia.org/wiki/Logit), `logit(x)=log(x/(1-x))` component-wise for `x` with entries between 0 and 1 (inclusive).
This is the inverse of the sigmoid function, and can be used to turn a probability into a score (the log odds) between `-inf` and `inf`.
The logit function is currently not implemented as a native TensorFlow op (a NumPy ndarray ufunc version exists in SciPy as [`scipy.special.logit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logit.html)). 

## Usage

Clone this repo and run `make`:

```bash
git clone https://github.com/artemmavrin/TF-Custom-CXX-Op.git
cd TF-Custom-CXX-Op
# Optional but recommended: create and activate a new Python virtual environment
make
```

Running `make` will download the required Python dependencies, install the `tf_custom_cxx_op` Python package, and compile the C++ op.
It will also run the unit tests.
These steps can be done separately using `make install` (to install dependencies and the Python package), `make build` (to compile the C++ op), and `make test` (to run the unit tests).

Using the `logit` TensorFlow op in Python is then easy.
For example, in a Python REPL:

```python
>>> from tf_custom_cxx_op import logit
>>> logit(0.5)  # Accepts a tensor of any shape
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
```

## TODO

- [ ] Add documentation.
- [x] Test on macOS.
- [x] Test on Linux.
- [ ] Test on Windows.
- [x] Create pip-installable package.
- [ ] Register gradient directly in C++.
- [ ] Make a GPU op.

## Acknowledgements

Good reading about creating TensorFlow C++ ops:

* The TensorFlow [Create an op](https://www.tensorflow.org/guide/create_op) guide.
* The [tensorflow/custom-op](https://github.com/tensorflow/custom-op) repo.
* David Stutz's blog post [Implementing Tensorflow Operations in C++ — Including Gradients](http://davidstutz.de/implementing-tensorflow-operations-in-c-including-gradients/) and [accompanying repo](https://github.com/davidstutz/tensorflow-cpp-op-example).
