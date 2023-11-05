# DISCLAIMER: the code is not working properly! If you still consider using, be prepated to debug it thoroughly.
Reproduction of AlphaTensor paper for 2x2 matrices. Parts of code are inspired by [this repo](https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py), but strongly refactored.

## Requirements

An optional first step, which will make everything easier
```
conda create --name alphastrassen python=3.8
conda activate alphastrassen
```

Install [torch](https://pytorch.org/) compatible with your CUDA version
```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch  # for CUDA >=11.3
```

Then, install our project
```
git clone https://github.com/migonch/alphastrassen.git
cd alphastrassen
pip install -e .
```
