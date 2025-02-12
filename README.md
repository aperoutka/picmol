
# PICMOL: Phase Instability Calculator for Molecular Design

## Features

- fragments molecules into UNIFAC subgroups
- support for UNIFAC, UNIQUAC, NRTL, and Flory-Huggins thermodynamic models
    - phase diagram calculations    
    - I(q=0) calculations from small angle X-ray scattering

## Environment Variables

To run this project, create an anaconda environment from environment.yml

```python
conda env create --name picmol -f environment.yml
```


## Examples

Example of using Tc_search -- finds the critical temperature for a given mixture.

```python
from chameleon import Tc_search

# example for using smiles identifier, does not require entry to by located in molecule_smiles.csv
Tc = Tc_search(Tmax=300, identifier=['CCO', 'CCCCCC'], identifier_type="smiles")

# example using the name identifier, molecule name and smiles must be located in molecule_smiles.csv
Tc = Tc_search(Tmax=300, identifier=['dbp', 'hexane'], identifier_type="name")

```

Using UNIFAC to calculate mixing free energy and derivatives, phase diagrams, and small-angle X-ray scattering I0.

```python
from chameleon import UNIFAC, thermo_analysis, make_plots

save_dir = f"os.getcwd()/figures/"

# perform UNIFAC analysis of DOHE/dodecane binary system
results = thermo_analysis(dz=0.001, Tmin=150, Tmax=300, dT=2, molecule_names=['dohe','dodecane'], thermo_model=UNIFAC)

# construct mixing free energy figures
make_plots(results=results, basis='vol', which_plots=['gmix'], save_dir=save_dir)

# construct scattering Io as a function of temperature and composition
make_plots(results=results, basis='vol', which_plots=['i0'], save_dir=save_dir, ymin=0.05, ymax=0.2)

# construct phase diagram (no heatmap)
make_plots(results=results, basis='vol', which_plots=['phase'], save_dir=save_dir)

# construct phase diagram with Gmix as heatmap z-axis
make_plots(results=results, basis='vol', which_plots=['phase gmix'], save_dir=save_dir)

# construct phase diagram with Io as heatmap z-axis
make_plots(results=results, basis='vol', which_plots=['phase i0'], save_dir=save_dir, ymin=0.05, ymax=1)

```

A collection of example jupyter notebooks are provided in the examples directory.


## Authors

- [Allison Peroutka](https://www.github.com/aperoutka)

