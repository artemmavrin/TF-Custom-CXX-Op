# Python setup
PYTHON = python
PIP = $(PYTHON) -m pip
PIP_INSTALL = $(PIP) install -U

# C++ setup
TF_CFLAGS = $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic -shared -undefined dynamic_lookup -fPIC $(TF_CFLAGS) -O2


.PHONY: all clean cxx_info py_info tensorflow test

all: clean py_info tensorflow cxx_info _logit.so test

_logit.so: logit.cc
	$(CXX) $(CXXFLAGS) -o $@ $^

test: _logit.so
	$(PYTHON) -m unittest discover --pattern '*test.py' --verbose

tensorflow: py_info
	$(PIP_INSTALL) pip setuptools wheel
	$(PIP_INSTALL) -r requirements.txt
	@ echo "Using TensorFlow $$($(PYTHON) -c 'import tensorflow as tf; print(tf.__version__)')"

py_info:
	@ echo "Using $$($(PYTHON) --version) at $$(which $(PYTHON))"

cxx_info:
	$(CXX) --version

clean:
	rm -f _logit.so
