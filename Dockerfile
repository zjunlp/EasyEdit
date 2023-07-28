############################################################
# Dockerfile to build KnowLM container images
# Based on Ubuntu
############################################################

# Use the official Ubuntu 20.04 image as your parent image.
FROM ubuntu:22.04

# Set the working directory within your container to /app.
WORKDIR /app

# Let the python output directly show in the terminal without buffering it first.
ENV PYTHONUNBUFFERED=1

# Update the list of packages, then install some necessary dependencies.
RUN apt-get update && apt-get install -y \
  wget \
  git \
  bzip2 \
  libglib2.0-0 \
  libxext6 \
  libsm6 \
  libxrender1 \
  make\
  g++ 

RUN rm -rf /var/lib/apt/lists/*

# Download and install the latest version of Miniconda to /opt/conda.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
  && rm Miniconda3-latest-Linux-x86_64.sh 

# Add Miniconda's binary directory to PATH.
ENV PATH /opt/conda/bin:$PATH

# Use conda to create a new environment named zhixi and install Python 3.9.
RUN conda create -n zhixi python=3.9 -y

# Initialize bash shell so that 'conda activate' can be used immediately.
RUN conda init bash

# Activate the conda environment.
RUN echo "conda activate zhixi" >> ~/.bashrc
ENV PATH /opt/conda/envs/zhixi/bin:$PATH

# Clone the zhixi project from GitHub.
RUN git clone https://github.com/zjunlp/KnowLM.git

# Change the working directory to the newly cloned zhixi project directory.
WORKDIR /app/KnowLM

# Activate the zhixi conda environment and install the Python dependencies listed in requirements.txt.
RUN /bin/bash -c "source ~/.bashrc && pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
RUN /bin/bash -c "source ~/.bashrc && pip install -r requirements.txt"

