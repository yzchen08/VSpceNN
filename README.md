## Welcome to VSpecNN page.

VSpecNN is a highly efficient multi-task ML surrogate model to accurately calculate infrared (IR) and Raman spectra 
based on dipole moments and polarizabilities obtained on-the-fly via ML-enhanced molecular dynamics (MD) simulations. 
For more details about VSpecNN, please visit https://arxiv.org/abs/2402.06911.

### environment preparation.

The environment for training is controlled by [anaconda](https://www.anaconda.com/). After installation of anaconda, 
simply run:
```commandline
conda env create -f environ.yaml
```

### Data preparation.

In order to train VSpecNN from scratch, the users need to prepare their own dataset 
(coordinates, energies, gradients, dipoles and polarizablities) for training. We have prepared demo 
dataset in `./data/train`, in which contains 5 files:
1. xyz.dat: coordinates in extxyz format in (n_moles * n_atoms, 3) (Ångström)
2. grad.dat: gradients (n_moles * n_atoms, 3) (Hartree/Bohr)
3. energy.dat: energeis (n_moles, 1) (kcal/mol)
4. dipole.dat: dipoles (n_moles, 3) (a.u.)
5. polarizability.dat: polarizabilities (n_moles, 9) (a.u.)

Once the above 5 files exist, run `gen_data.py train` in `./data` folder to generate a `data.pt` file.

### Model training.

After preparation of `data.pt`, run `run_total.py` in `VSpecNN` folder, the best model will be saved in `log` folder.
