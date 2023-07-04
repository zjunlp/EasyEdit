#!/bin/bash

# Start from directory of script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# Detect operating system
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

if [ $machine != "Linux" ] && [ $machine != "Mac" ]
then
	echo "Conda setup script is only available on Linux and Mac"
	exit 1
else
	echo "Running on $machine"
fi

# Default RECIPE 'rome' can be overridden by 'RECIPE' environment variable
RECIPE=${RECIPE:-rome}
# Default ENV_NAME 'rome' can be overridden by 'ENV_NAME'
ENV_NAME="${ENV_NAME:-${RECIPE}}"
echo "Creating conda environment ${ENV_NAME}"

if [[ ! $(type -P conda) ]]
then
    echo "conda not in PATH"
    echo "read: https://conda.io/docs/user-guide/install/index.html"
    exit 1
fi

if df "${HOME}/.conda" --type=afs > /dev/null 2>&1
then
    echo "Not installing: your ~/.conda directory is on AFS."
    echo "Use 'ln -s /some/nfs/dir ~/.conda' to avoid using up your AFS quota."
    exit 1
fi

CUDA_DIR="/usr/local/cuda-11.1"
if [[ ! -d ${CUDA_DIR} ]]
then
    echo "Environment requires ${CUDA_DIR}, not found."
    exit 1
fi

# Uninstall existing environment
# conda deactivate
rm -rf ~/.conda/envs/${ENV_NAME}

# Build new environment: torch and torch vision from source
# CUDA_HOME is needed
# https://github.com/rusty1s/pytorch_scatter/issues/19#issuecomment-449735614
conda env create --name=${ENV_NAME} -f ${RECIPE}.yml

# Set up CUDA_HOME to set itself up correctly on every source activate
# https://stackoverflow.com/questions/31598963
mkdir -p ~/.conda/envs/${ENV_NAME}/etc/conda/activate.d
echo "export CUDA_HOME=${CUDA_DIR}" > \
    ~/.conda/envs/${ENV_NAME}/etc/conda/activate.d/CUDA_HOME.sh

# conda activate ${ENV_NAME}
