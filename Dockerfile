# Use the official Ubuntu 22.04 image
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Set non-interactive mode to avoid issues during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary dependencies in a single RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
  wget \
  git \
  bzip2 \
  libglib2.0-0 \
  libxext6 \
  libsm6 \
  libxrender1 \
  make \
  g++ \
  && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
  && rm Miniconda3-latest-Linux-x86_64.sh 

# Set PATH to include Miniconda
ENV PATH="/opt/conda/bin:$PATH"

# Clone the EasyEdit project
RUN git clone https://github.com/zjunlp/EasyEdit.git && \
    cd EasyEdit

# Copy environment file and create the Conda environment
COPY environment.yml /app/EasyEdit/
RUN conda env create -f /app/EasyEdit/environment.yml

# Use conda shell for all subsequent commands
SHELL ["conda", "run", "-n", "EasyEdit", "/bin/bash", "-c"]

# Set Conda default environment
RUN echo "source activate EasyEdit" > ~/.bashrc
ENV PATH="/opt/conda/envs/EasyEdit/bin:$PATH"

# Install additional dependencies
COPY requirements.txt /app/EasyEdit/
RUN pip install --no-cache-dir -r /app/EasyEdit/requirements.txt

# Set working directory
WORKDIR /app/EasyEdit

# Expose any required ports (e.g., Jupyter Notebook)
EXPOSE 8888

# Default command
CMD ["bash"]
