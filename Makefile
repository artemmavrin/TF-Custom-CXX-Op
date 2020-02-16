# Python setup
PYTHON = python
PIP = $(PYTHON) -m pip
PIP_INSTALL = $(PIP) install -U

# C++ setup
TF_CFLAGS = $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS = $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2
ifneq ($(OS), Windows_NT)
  ifeq ($(shell uname -s), Darwin)
    CXXFLAGS += -undefined dynamic_lookup
  endif
endif

LIBRARIES = _logit.so

.PHONY: all clean cxx_info py_info test tf_info

all: clean tf_info cxx_info test

_%.so: %.cc
	$(CXX) $< -o $@ $(CXXFLAGS)

test: $(LIBRARIES)
	$(PYTHON) -m unittest discover --pattern '*test.py' --verbose

tf_info: py_info
	$(PIP_INSTALL) pip setuptools wheel
	$(PIP_INSTALL) -r requirements.txt
	@ echo "Using TensorFlow $$($(PYTHON) -c 'import tensorflow as tf; print(tf.__version__)')"

py_info:
	@ echo "Using $$($(PYTHON) --version) at $$(which $(PYTHON))"

cxx_info:
	$(CXX) --version

clean:
	rm -f *.so
