[![Documentation Status](https://readthedocs.org/projects/mlcf-master/badge/?version=latest)](https://mlcf-master.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/semodi/mlcf.svg?branch=master)](https://travis-ci.org/semodi/mlcf)

![model](https://github.com/semodi/mlcf/modle.png)

# Machine learned correcting functionals (MLCF)

This repository provides an implementation of the MLCF method introduced in https://arxiv.org/abs/1812.06572. 
MLCFs add a layer on top density functional theory (DFT) calculations that correct force and energy predictions. To do so they use a representation of the real-space electron density as input. This implementation provides a calculator class that is designed to work together with the [Atomic Simulation Environment (ase)](https://wiki.fysik.dtu.dk/ase/).

It is divided into three sub-modules:

### mlcf_func.elf
 
This submodule takes care of the electron density representation (elf stands for **El**ectronic **F**ingerprints). In detail, this means that the electron density is read from a ASCII file whose formatting is specific to the underlying DFT method. Further, the electron density is projected onto a set of basis sets and then rotated into a atomic local coordinate system to ensure that the MLCF transforms covariantly under rotations.

### mlc_func.ml


