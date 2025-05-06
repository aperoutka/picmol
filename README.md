# picmol: Phase Instability Calculator for MOLecular design

`picmol` is a `Python` library designed to streamline the analysis of molecular dynamics simulations, focusing on the application of Kirkwood-Buff (KB) theory for the calculation of activity coefficients, excess thermodynamic properties, and liquid-liquid equilibria (LLE).

[![docs](http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://picmol.readthedocs.io/)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://tldrlegal.com/license/mit-license)
![python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)


## Key Features

* **Kirkwood-Buff Integration:**
    * Calculates activity coefficients and excess thermodynamic properties from molecular dynamics simulations using Kirkwood-Buff theory.
    * Fits interaction parameters to various thermodynamic models, including:
        * Numerical (4th order Taylor series expansion)
        * UNIQUAC
        * NRTL
        * Flory-Huggins
    * Includes support for predictive thermodynamic models, including:
        * UNIFAC
      
* **Liquid-Liquid Equilibria (LLE) Calculations:**
    * Performs phase diagram calculations for LLE systems, leveraging the KB-derived interaction parameters.
    * Generates phase diagrams for visual analysis.
      
* **Small-Angle X-ray Scattering (SAXS) Support:**
    * Includes functionality to calculate SAXS Io, providing valuable structural information and comparison to experimental techniques.
      
* **Visualization Tools:**
    * Offers comprehensive visualization capabilities for both Kirkwood-Buff analysis results and LLE phase diagrams, aiding in data interpretation.

## Installation

`picmol` can be installed from cloning github repository.

```python
git clone https://github.com/aperoutka/picmol.git
```

Creating an anaconda environment with dependencies and installation.

```python
cd picmol
conda env create -f environment.yml
conda activate picmol
pip install .
```

## File Structure Requirements

To use the Picmol package, your project and pure component directories should be structured as follows:

```markdown
kbi_dir/
├── project/
│   └── system/
│       ├── rdf_dir/
│       │   ├── mol1_mol1.xvg
│       │   ├── mol1_mol2.xvg
│       │   └── mol1_mol2.xvg
│       ├── system_npt.edr
│       └── system.top
└── pure_components/
    └── molecule_temp/
        ├── molecule_temp_npt.edr
        └── molecule_temp.top
```

**Explanation:**

* **`kbi_dir/`**: Parent directory for KBI analysis that contains subdirectories of projects and pure components.
* **`project_directory/`**: This is the root directory for your `picmol` project and contains system subdirectories (**`system1/`, `system2/`, etc.:**) with various compositions.
* **`rdf_dir/`**: Inside each system directory, you must have an `rdf_dir` subdirectory. This directory should contain all the radial distribution function `.xvg` files that you want to analyze.
* **`.edr` file**: Each system directory must also contain the `.edr` file from the NPT production run of your simulation, should have `npt` in its name.
* **`.top` file**: Each system directory must contain the `.top` topology file with system information.
* **`pure_components`**: directory containing all the pure components, where the system names should be [molecule_name]_[temp].

**Important Notes:**

* Ensure that the `.edr`, `.top`, and `.xvg` files are correctly placed within their respective system directories.
* RDF directory name should be the same for each system.
* The naming convention of your RDF files (`rdf1.xvg`, `rdf2.xvg`, etc.) is flexible, but ensure molecular names (same as in `.top` file) are in filename.
* `picmol` expects this specific file structure to function correctly.

## Examples

Detailed Juptyer notebook examples are provided in the [docs](https://picmol.readthedocs.io/), with select examples of code cells and command line tools provided below.

`picmol` includes support for a command line interface, for rdf generation and kbi analysis. To print a list of all arguments and their default values for RDF and KBI analysis.

```python
picmol-rdf -h
```

```python
picmol-kbi -h
```

Alternatively, this could be run with `Python`.

```python
from picmol import KBI, KBIPlotter, ThermoModel, PhaseDiagramPlotter

# performs the kbi analysis, including calculating activity coefficients, excess thermodynamic properties, and fitting thermodyanmic model interaction parameters.
# kbi_method, options: "adj", "kgv", "raw"
kbi_obj = KBI(prj_path, pure_component_path, solute_mol, rdf_dir, kbi_method, avg_start_time, avg_end_time, kbi_fig_dirname)
kbi_obj.kbi_analysis()

# make figures
kbi_plotter = KBIPlotter(kbi_obj)
kbi_plotter.make_figures()

# for creating a numerical thermodynamic model
# model_name, options: "quartic", "uniquac", "unifac", "fh", "nrtl"
tmodel = ThermoModel(model_name="quartic", KBIModel=kbi_obj, dT=1, Tmin=200, Tmax=400)
tmodel.temperature_scaling()

# create LLE figures
tmodel_plotter = PhaseDiagramPlotter(tmodel)
tmodel_plotter.make_figures()
```

Predicting LLE with UNIFAC, doesn't require KB model because its a predictive method. 

```python
from picmol import load_molecular_properties

# load molecular properties
molec_info = load_molecular_properties('mol_name')

# get smiles of molecules
mols = ['DOHE', 'HEPTANE']
smiles = molec_info.loc[mols,:]['smiles'].tolist()

# create unifac thermodynamic model
tmodel = ThermoModel(identifiers=smiles, identifier_type="smiles", model_name="unifac", unif_version="unifac", dT=5, Tmin=200, Tmax=400)
tmodel.temperature_scaling()
```

Estimating critical temperatures for a binary system with UNIFAC.

```python
from picmol import Tc_search, load_molecular_properties

# load molecular properties
molec_info = load_molecular_properties('mol_name')

# get smiles of molecules
mols = ['DOHE', 'HEPTANE']
smiles = molec_info.loc[mols,:]['smiles'].tolist()

# get Tc for mixture
# lle_type, options = "ucst", "lcst", None
Tc = Tc_search(Tmax=500, smiles=smiles, lle_type="ucst")
```

To add a molecule to `molecular_properties.csv`.

```python
from picmol import add_molecule

"""
mol_name: molecule name
mol_id: molecule id in simulation .top file
density: liquid density (g/mL) at 298.15 K
smiles: SMILES string for RDkit interfacing and UNIFAC modeling

Default is to calculate molar volume from experimental mass density
If density is not entered, molar volume will be calculated from RDkit and the mass density will be calculated from molar volume
"""

add_molecule(mol_name, mol_id, smiles, density)
```
