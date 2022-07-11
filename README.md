# Reducing Certified Regression to Certified Classification

[![docs](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ZaydH/certified-regression/blob/main/LICENSE)

This repository contains the source code for reproducing the results in the paper "Reducing Certified Regression to Certified Classification".

* Authors: [Zayd Hammoudeh](https://zaydh.github.io/) and [Daniel Lowd](https://ix.cs.uoregon.edu/~lowd/)
* Link to Paper: [Arxiv](https://arxiv.org/abs/2208.13904)

## Running the Program

To run the program, enter the `src` directory and call:

`python driver.py ConfigFile`

where `ConfigFile` is one of the `yaml` configuration files in folder [`src/configs`](src/configs). 

* To disable weighted mode, run the program with the flag `--no_multi`
* To disable overlapping mode, run the program with the flag `--deg 1`
* To override threshold $\xi$, run the program with the flag `--dist c` where $\xi = c$. For the datasets `ames_housing`, `austin_housing`, and `diamonds`, `c` is denotes a percentage of true target value $y_{\text{te}}$. For datasets `weather` and `life`, `c` is a simple scalar.

### First Time Running the Program

The first time each configuration runs, the program automatically downloads any necessary dataset(s).  Please note that this process can be time-consuming -- in particular for the `weather` dataset.

These downloaded files are stored in a folder `.data` that is in the same directory as `driver.py`.  If the program crashes while running a configuration for the first time, we recommend deleting or moving the `.data` to allow the program to re-download and reinitialize the source data.

### Gurobi License

By default, this program loads `gurobipy`, [Gurobi's python package](https://pypi.org/project/gurobipy/). For non-trivial linear programs, `gurobipy` requires a license.  Academic users can [request free, unlimited licenses](https://www.gurobi.com/academia/academic-program-and-licenses/) directly from Gurobi.

### Requirements

Our implementation was tested in Python&nbsp;3.7.1.  For the full requirements, see `requirements.txt` in the `src` directory.  If a different version of Python is used, some package settings in `requirements.txt` may need to change.

We recommend running our program in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).  Once your virtual environment is created and *active*, run the following in the `src` directory:

```
pip install --user --upgrade pip
pip install -r requirements.txt
```

### License

[MIT](https://github.com/ZaydH/certified-regression/blob/main/LICENSE)

## Citation

```
@misc{Hammoudeh:2022:CertifiedRegression,
    author = {Hammoudeh, Zayd and
              Lowd, Daniel},
    title = {Reducing Certified Regression to Certified Classification},
    journal   = {CoRR},
    year      = {2022},
    eprint = {2208.13904},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
}
```
