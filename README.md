# BASIC

Baseline Aerodynamic Solver for Ideal Compressible flows

Reference : Volpiani, P.S. A Comprehensive Study About Implicit/Explicit Large-Eddy Simulations with Implicit/Explicit Filtering. Flow Turbulence Combust (2024). https://doi.org/10.1007/s10494-024-00577-9


## Getting started

### Instal python libraries using conda [https://www.anaconda.com/] and create basic environment

conda create -n basic python=3.11.4
conda activate basic
conda install numpy=1.25.0
conda install matplotlib=3.7.1
conda install mpi4py=3.1.4
conda install pyyaml=6.0
conda install scipy 
conda install pytorch
conda install scikit-learn
conda install pandas
pip install pyvista

### Running a case

conda activate basic
mpiexec -n 2 python main.py
