## PLUE-SFRDA-Pseudo Label Uncertainty Estimation-Source Free Robust Domain Adaptation
# Prerequisites
> - Ubuntu 18.04
> - Python 3.6+
> - PyTorch 1.5+ (recent version is recommended)
> - NVIDIAGPU (>=11GB)
> - CUDA 10.0 (optional)
> - CUDNN7.5 (optional)

# Getting Started
# Installation.
 - Install python libraries\
 >conda install -c conda-forge matplotlib\
 >conda install -c anaconda yaml\
 >conda install -c anaconda pyyaml\
 >conda install -c anaconda scipy\
 >conda install -c anaconda scikit-learn\
 >conda install -c conda-forge easydict\
 >pip insatll easydl
# Download source-pretrained parameters (Fs and Cs of Figure 2 in our main paper)
- Download source-pretrained parameters[[link]](https://pan.baidu.com/s/1r7YDhhdpXBanbpNEnIkxvA),password: wfff ;in: save_model_path:replace\
- ex) source-pretrained parameters of A[0] -> W[2] senario should be located in /pretrained_weights_office31/TrainSourceModelaccBEST_model_checkpoint01.pth.tar
# Training
- Run the following command
 ```python
  python SFDA_train.py
  ```
 

