# picmol: Phase Instability Calculator for MOLecular design

`picmol` is a `Python` library designed to streamline the analysis of molecular dynamics simulations, focusing on the application of Kirkwood-Buff (KB) theory for the calculation of activity coefficients, excess thermodynamic properties, and liquid-liquid equilibria (LLE).

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://tldrlegal.com/license/mit-license)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)


## Key Features

* **Kirkwood-Buff Integration:**
    * Calculates activity coefficients and excess thermodynamic properties from molecular dynamics simulations using Kirkwood-Buff theory.
    * Fits interaction parameters to various thermodynamic models, including:
        * Numerical models (e.g., quartic)
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

## Workflow

`picmol` provides a cohesive workflow that bridges the gap between molecular dynamics simulations and phase equilibria calculations. It enables users to:

1.  **Analyze MD simulations:** Use Kirkwood-Buff theory to derive excess thermodynamic properties.
2.  **Fit thermodynamic models:** Fit interaction parameters to suitable thermodynamic models.
3.  **Calculate LLE:** Predict liquid-liquid equilibria and generate phase diagrams.
4.  **Analyze SAXS data:** Derive SAXS Io values from free energy curvature.
5.  **Visualize results:** Generate informative plots for analysis and presentation.

## Installation

`picmol` can be installed from cloning github repository.

```python
git clone https://github.com/aperoutka/picmol.git
```
Creating an anaconda environment with `picmol` dependencies.

```python
conda env create -f environment.yml
```
Installing picmol package.

```python
pip install .
```

## File Structure Requirements

To use the Picmol package, your project directory should be structured as follows:

**Explanation:**

* **`project_directory/`**: This is the root directory for your `picmol` project.
* **`system1/`, `system2/`, etc.:** Each subdirectory within the `project_directory` represents a separate simulation system you want to analyze.
* **`rdf/`**: Inside each system directory, you must have an `rdf` subdirectory. This directory should contain all the radial distribution function (RDF) `.xvg` files that you want to analyze with `picmol`.
* **`.edr` file**: Each system directory must also contain the `.edr` file from the NPT production run of your simulation.
* **`.top` file**: Similarly, each system directory must contain the `.top` topology file corresponding to the system.

**Important Notes:**

* Ensure that the `.edr`, `.top`, and `.xvg` files are correctly placed within their respective system directories.
* RDF directory name should be the same for each system.
* The naming convention of your RDF files (`rdf1.xvg`, `rdf2.xvg`, etc.) is flexible, but ensure molecular names (same as in `.top` file) are in filename.
* `picmol` expects this specific file structure to function correctly.

## Examples

`picmol` includes support for a command line interface, and example is provided below where the only argument different from default values is the name for rdf file directory.

```python
python -m picmol.kbi_analysis --rdf_dir rdf_files_npt
```

To print a list of all arguments and their default values, use the following.

```python
python -m picmol.kbi_analysis -h
```

Alternatively, this could be run from inside a python script.

```python
from picmol import KBI, KBIPlotter, ThermoModel, PhaseDiagramPlotter

# performs the kbi analysis, including calculating activity coefficients, excess thermodynamic properties, and fitting thermodyanmic model interaction parameters.
# kbi_method, options: "adj", "kgv", "raw"
kbi_obj = KBI(prj_path, pure_component_path, rdf_dir, kbi_method, avg_start_time, avg_end_time, kbi_fig_dirname)
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
mol_class: "extractant", "modifier", "solute", or "solvent"
smiles: SMILES string for RDkit interfacing and UNIFAC modeling

Default is to calculate molar volume from experimental mass density
If density is not entered, molar volume will be calculated from RDkit and the mass density will be calculated from molar volume
"""
add_molecule(mol_name, mol_id, density, mol_class, smiles)
```








