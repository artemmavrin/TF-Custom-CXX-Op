FROM tensorflow/tensorflow:latest-py3
ENV HOME /home/tf
RUN mkdir -p $HOME
WORKDIR $HOME
COPY . $HOME
RUN make install
RUN make build
CMD make test
