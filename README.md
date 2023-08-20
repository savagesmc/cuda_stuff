# Overview

This is my sandbox repo for learning cuda programming.

## Links

- [cuda-toolkit-archive - download site](https://developer.nvidia.com/cuda-toolkit-archive)
- [CUDA Toolkit Documentation (web)](https://docs.nvidia.com/cuda/)
- [CUDA Toolkit Reference Manual (pdf)](https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_Toolkit_Reference_Manual.pdf)
- [CUDA C++ Programmers Guide (pdf)](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
- [CUDA C++ Best Practices (pdf)](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)
- [CUDA Runtime API Reference Manual (pdf)](https://docs.nvidia.com/cuda/pdf/CUDA_Runtime_API.pdf)

## Install Instructions

### Ubuntu 22.04

``` bash
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt install nvidia-cuda-toolkit
```

#### Cuda Toolkit 11.8 (First version supported in 22.04)

``` bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### PyTorch
