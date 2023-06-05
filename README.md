# A study of Vectorial Total Generalized Variation reconstruction for undersampled CT

![comparison](/figs/comparison.png)

This is a project conducted within the scope of 227-0424-00L at ETH Zürich during spring 2023.

[This Jupyter notebook](tgv-recon.ipynb) implements the following:

- Vectorial Total Generalized Variation reconstructor using ADMM
- Total Variation reconstructor using ADMM
- Generalized Tikhonov reconstructor using Conjugate gradients

Furthermore, there is an interface for using the built-in Filtered backprojection algorithm from the [scikit-image package](https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html#reconstruction-with-the-filtered-back-projection-fbp).

Some needed functionality for the notebook are located in the [utils.py](utils.py) file. This file implements:

- loading data from file
- creation of radon matrices
- generation of undersampling matrix (implemented as a vector for faster computation)
- $\ell_{2,1}$-norm
- forward simulation of CT data from MRI image
- generation of sparse gradient matrices (not in use!)

## Installation
All required packages are listed in [environment.yml](environment.yml). To install them using Conda, run ```conda env create -f environment.yml```. This will create a virtual enviroment called ```tgv-recon``` on your machine.

## The paper
The study is discussed and summarized [here](paper.pdf).
