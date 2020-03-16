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
PYTHON_DIR = src/python/tf_custom_cxx_op
CXX_DIR = src/cc
LOGIT_TARGET_LIB = $(PYTHON_DIR)/_logit.so
LOGIT_SRCS = $(CXX_DIR)/logit.cc

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
	$(PYTHON) -m unittest discover ./tests --verbose

py_info:
	@ echo "Using $$($(PYTHON) --version) at $$(which $(PYTHON))"

clean:
	rm -f $(PYTHON_DIR)/*.so

docker-build:
	$(DOCKER_BUILD) $(DOCKER_IMAGE) .

docker-test: docker-build
	$(DOCKER_RUN) $(DOCKER_CONTAINER) $(DOCKER_IMAGE)

docker-run: docker-build
	$(DOCKER_RUN) $(DOCKER_CONTAINER) --entrypoint /bin/bash $(DOCKER_IMAGE) -i
