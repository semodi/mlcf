[![Documentation Status](https://readthedocs.org/projects/mlcf-master/badge/?version=latest)](https://mlcf-master.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/semodi/mlcf.svg?branch=master)](https://travis-ci.org/semodi/mlcf)

<img src="https://github.com/semodi/mlcf/blob/master/model.png" width="400" height="210" />

# Machine learned correcting functionals (MLCF)

This repository provides an implementation of the MLCF method introduced in https://arxiv.org/abs/1812.06572. 
MLCFs add a layer on top density functional theory (DFT) calculations to correct force and energy predictions. To do so they use a representation of the real-space electron density as input. This implementation provides a calculator class that is designed to work together seamlessly with the [Atomic Simulation Environment (ase)](https://wiki.fysik.dtu.dk/ase/).

## Installation

To generate the Wigner-D matrices used to rotate electronic descriptors the following package needs to be installed:

`conda install -c moble spherical_functions`

Afterwards, mlc_func can be installed by cloning this directory as follows:

```
git clone https://github.com/semodi/mlcf.git
cd mlcf
pip install -e .
```

## Modules

`mlc_func` is divided into three submodules:

### mlcf_func.elf
 
This submodule takes care of the electron density representation (elf stands for **El**ectronic **F**ingerprints). In detail, this means that the electron density is read from an ASCII file whose formatting is specific to the underlying DFT method. Further, the electron density is projected onto a set of basis sets. The resulting descriptors are subsequently rotated into atomic local coordinate systems to ensure that the MLCF transforms covariantly under rotations.

### mlc_func.ml

All routines concerning machine learning are contained in this submodule. Reasonable default parameters are implemented, allowing the user to build MLCFs without any prior knowledge of machine learning and neural networks. 

### mlc_func.md

To use the trained MLCFs for structural relaxation, molecular dynamics etc., this submodule implements an ase.Calculator class. Inheriting from the ase.Calculator base class, MLCFs are able to harvest the entire functionality of the atomic simulation environment. Furthermore, any DFT calculator that is supported by ase can in theory be used as the baseline method for the MLCF, these calculators include Abinit, Quantum Espresso, Gaussian, Siesta and many others.





