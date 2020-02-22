# Python setup
PYTHON = python
PIP = $(PYTHON) -m pip
PIP_INSTALL = $(PIP) install --upgrade

# C++ setup
TF_CFLAGS = $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS = $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2
ifneq ($(OS), Windows_NT)
  ifeq ($(shell uname -s), Darwin)
    CXXFLAGS += -undefined dynamic_lookup
  endif
endif

# Project directories and files
PACKAGE_DIR = src/tf_custom_cxx_op
PYTHON_DIR = $(PACKAGE_DIR)/python
CXX_DIR = $(PACKAGE_DIR)/cc
LOGIT_TARGET_LIB = $(PYTHON_DIR)/ops/_logit_ops.so
LOGIT_SRCS = $(CXX_DIR)/kernels/logit_kernels.cc $(CXX_DIR)/ops/logit_ops.cc

# Docker setup
DOCKER_BUILD = docker build -t
DOCKER_RUN = docker run -it --rm --name
DOCKER_IMAGE = tf_custom_cxx_op
DOCKER_CONTAINER = tf_custom_cxx_op_test

.PHONY: all build clean docker-build docker-test py_info test

all: clean install build test

$(LOGIT_TARGET_LIB): $(LOGIT_SRCS)
	$(CXX) $^ -o $@ $(CXXFLAGS)

install: py_info
	$(PIP_INSTALL) --editable .

build: $(LOGIT_TARGET_LIB)

test:
	$(PYTHON) -m unittest discover $(PACKAGE_DIR) --verbose

py_info:
	@ echo "Using $$($(PYTHON) --version) at $$(which $(PYTHON))"

clean:
	rm -f $(PACKAGE_DIR)/python/ops/*.so

docker-build:
	$(DOCKER_BUILD) $(DOCKER_IMAGE) .

docker-test: docker-build
	$(DOCKER_RUN) $(DOCKER_CONTAINER) $(DOCKER_IMAGE)

docker-run: docker-build
	$(DOCKER_RUN) $(DOCKER_CONTAINER) --entrypoint /bin/bash $(DOCKER_IMAGE) -i
