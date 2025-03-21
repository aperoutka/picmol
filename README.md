
# PICMOL: PHASE INSTABILITY CALCULATOR FOR MOLECULAR DESIGN

## Features

- fragments molecules into UNIFAC subgroups
- support for UNIFAC, UNIQUAC, NRTL, and Flory-Huggins thermodynamic models
    - phase diagram calculations    
    - I(q=0) calculations from small angle X-ray scattering

## Environment Variables

To run this project, create an anaconda environment from environment.yml

```python
conda env create -f environment.yml
```


## Examples

Example of using Tc_search -- finds the critical temperature for a given mixture.

```python
from picmol import Tc_search

# example for using smiles identifier, does not require entry to by located in molecule_smiles.csv
Tc = Tc_search(Tmax=300, identifier=['CCO', 'CCCCCC'], identifier_type="smiles")

# example using the name identifier, molecule name and smiles must be located in molecule_smiles.csv
Tc = Tc_search(Tmax=300, identifier=['dbp', 'hexane'], identifier_type="name")

```


A collection of example jupyter notebooks are provided in the examples directory.


## Authors

- [Allison Peroutka](https://www.github.com/aperoutka)

