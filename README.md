# TensorFlow Custom C++ Op


![Last Commit](https://img.shields.io/github/last-commit/artemmavrin/TF-Custom-CXX-Op/master "Last Commit")
![License](https://img.shields.io/github/license/artemmavrin/TF-Custom-CXX-Op "License")

**This Repo Is Under Construction**

This is an example of how to create a custom TensorFlow op in C++ and use it in Python.

The example op implemented here is the [logit function](https://en.wikipedia.org/wiki/Logit), which is the inverse of the sigmoid function.
In other words, `logit(x) = log(x / (1 - x))` for `x` between 0 and 1 (inclusive).
This function exists in SciPy as [`scipy.special.logit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logit.html), but it is currently not implemented as a native TensorFlow op.

## Usage

Clone this repo and run `make`:

```bash
git clone https://github.com/artemmavrin/TF-Custom-CXX-Op.git
cd TF-Custom-CXX-Op
# Optional but recommended: create and activate a new Python virtual environment
make
```

Running `make` will download the required Python packages and compile the `_logit.so` shared library file, which is loaded by [`logit.py`](logit.py). It will also run the unit tests in [`logit_test.py`](logit_test.py).

Using the `logit` TensorFlow op in Python is then easy:

```python
>>> from logit import logit
>>> logit(0.5)  # Accepts a tensor of any shape
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
```

This was tested to work on macOS 10.

## TODO

- [ ] Add documentation.
- [ ] Create pip-installable package.
- [ ] Register gradient directly in C++.
- [ ] Make a GPU op.
