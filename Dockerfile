FROM nvidia/cuda:8.0-cudnn6-devel

MAINTAINER Nathan Douglas

##################################
# NOTE: To compile with GPU supply
# docker build args:

# --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
#
##################################
ARG dlib_build_args

ENV PYTHONPATH=/usr/src/app:$PYTHONPATH

#############################
# Install Ubuntu Dependencies
#############################
RUN apt-get update
RUN apt-get install -y git python-numpy python-dev python-pip wget locales vim telnet

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

#############################
# Install conda
#############################
RUN wget -q "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
RUN bash Mini*.sh -b -p /root/.anaconda
ENV PATH=/root/.anaconda/bin:$PATH


#############################
# Install Conda Packages
#############################
RUN conda update conda

RUN conda install -qy boost=1.61.0
RUN conda install -qyc menpo opencv
RUN conda install -qy h5py
RUN conda install -qy cmake

############################
# Build/Install Dlib
#############################
ENV DLIB_BUILD_ARGS=${dlib_build_args:-"--no DLIB_USE_CUDA"}

RUN git clone https://github.com/davisking/dlib.git
RUN cd dlib && python setup.py install ${DLIB_BUILD_ARGS}

####################################################
# Install faiss
####################################################
RUN apt-get update && \
    apt-get install -y libopenblas-dev 
    # apt-get install -y libopenblas-dev python-numpy python-dev swig git python-pip 

RUN pip install matplotlib

RUN cd /opt && git clone https://github.com/nathandouglas/faiss.git
# COPY . /opt/faiss

WORKDIR /opt/faiss

ENV BLASLDFLAGS /usr/lib/libopenblas.so.0

RUN mv example_makefiles/makefile.inc.Linux ./makefile.inc

# Set the proper flags to build for python3
RUN sed -i 's|\(PYTHONCFLAGS=\)\(.*$\)|\1-I /root/.anaconda/include/python3.6m -I /root/.anaconda/lib/python3.6/site-packages/numpy/core/include|g' ./makefile.inc

RUN make tests/test_blas -j $(nproc) && \
    make -j $(nproc) && \
    make demos/demo_sift1M -j $(nproc) && \
    make py

RUN cd gpu && \
    make -j $(nproc) && \
    make test/demo_ivfpq_indexing_gpu && \
    make py

ENV PYTHONPATH /opt/faiss:$PYTHONPATH

WORKDIR /
#############################
# Create placeholder
# Models will be 
# mounted to this location.
#############################
RUN mkdir -p $HOME/.tdesc/models/dlib
RUN mkdir -p $HOME/.keras/models

#############################
# Install Python Dependencies
#############################
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install --upgrade numpy

COPY . /tdesc

RUN cd /tdesc && python setup.py install 
