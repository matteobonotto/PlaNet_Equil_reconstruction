# PlaNet: reconstruction of plasma equilibrium and separatrix using convolutional physics-informed neural operator

Source code of [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0920379624000474).

Dataset available at [this repo](https://github.com/matteobonotto/ITERlike_equilibrium_dataset.git).



# Installation
First, create a virtual environment using `venv`
```shell
python3.10 -m venv venv 
source venv/bin/activate
```
then install the package and all the dependencies using `poetry`
```shell
pip3 install poetry==1.8.3
poetry config virtualenvs.create false
poetry install
```
For building the wheels
```shell
poetry build -f wheel
```