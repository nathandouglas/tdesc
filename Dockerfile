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
RUN apt-get install -y git python-numpy python-dev python-pip wget locales

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
RUN conda install -qy anaconda
RUN conda install -qy boost=1.61.0
RUN conda install -qy opencv
RUN conda install -qy menpo 
RUN conda install -qy h5py
RUN conda install -qy cmake

############################
# Build/Install Dlib
#############################
ENV DLIB_BUILD_ARGS=${dlib_build_args:-"--no DLIB_USE_CUDA"}

RUN git clone https://github.com/davisking/dlib.git
RUN cd dlib && python setup.py install ${DLIB_BUILD_ARGS}


####################################################
# Install darknet
####################################################
RUN git clone https://github.com/bkj/darknet && \
    cd darknet && mkdir build && cd build && \
    cmake .. && \
    make all -j8

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

COPY . /tdesc

RUN cd /tdesc && python setup.py install 
