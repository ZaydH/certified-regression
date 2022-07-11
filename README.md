# Certifiably Robust Regression Against Poisoning and Worst-Case Outliers

[![docs](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ZaydH/certified-regression/blob/main/LICENSE)

This repository contains the source code for reproducing the results in the paper "Certifiably Robust Regression Against Poisoning and Worst-Case Outliers".

## Running the Program

To run the program, enter the `src` directory and call:

`python driver.py ConfigFile`

where `ConfigFile` is one of the `yaml` configuration files in folder [`src/configs`](src/configs). 

* To disable weighted mode, run the problem with the flag `--no_multi`
* To disable overlapping mode, run the problem with the flag `--deg 1`

### First Time Running the Program

The first time each configuration is run, the program automatically downloads any necessary dataset(s).  Please note that this process can be time consuming --- in particular for the `weather` dataset.

These downloaded files are stored in a folder `.data` that is in the same directory as `driver.py`.  If the program crashes while running a configuration for the first time, we recommend deleting or moving the `.data` to allow the program to redownload and reinitialize the source data.

### Requirements

Our implementation was tested in Python&nbsp;3.7.1 but `requirements.txt` may need to change depending on your local Python configuration.  For the full requirements, see `requirements.txt` in the `src` directory.

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
    title = {Certifiably Robust Regression Against Poisoning and Worst-Case Outliers},
    year = {2022},
}
```
