# picmol: Molecular Simulation Analysis and Phase Equilibria Calculations

`picmol` is a `Python` library designed to streamline the analysis of molecular dynamics simulations, focusing on the application of Kirkwood-Buff (KB) theory for the calculation of activity coefficients, excess thermodynamic properties, and liquid-liquid equilibria (LLE).

## Key Features:

* **Kirkwood-Buff Integration:**
    * Calculates activity coefficients and excess thermodynamic properties from molecular dynamics simulations using Kirkwood-Buff theory.
    * Fits interaction parameters to various thermodynamic models, including:
        * Numerical models (e.g., quartic)
        * UNIQUAC
        * NRTL
        * Flory-Huggins
    * Includes support for predictive thermodynamic models, including:
        * UNIFAC
        * COSMO-RS
      
* **Liquid-Liquid Equilibria (LLE) Calculations:**
    * Performs phase diagram calculations for LLE systems, leveraging the KB-derived interaction parameters.
    * Generates phase diagrams for visual analysis.
      
* **Small-Angle X-ray Scattering (SAXS) Support:**
    * Includes functionality to calculate Io from SAXS data, providing valuable structural information.
      
* **Visualization Tools:**
    * Offers comprehensive visualization capabilities for both Kirkwood-Buff analysis results and LLE phase diagrams, aiding in data interpretation.

## Workflow:

`picmol` provides a cohesive workflow that bridges the gap between molecular dynamics simulations and phase equilibria calculations. It enables users to:

1.  **Analyze MD simulations:** Use Kirkwood-Buff theory to derive interaction parameters.
2.  **Fit thermodynamic models:** Fit the derived parameters to suitable thermodynamic models.
3.  **Calculate LLE:** Predict liquid-liquid equilibria and generate phase diagrams.
4.  **Analyze SAXS data:** Derive Io values from SAXS data.
5.  **Visualize results:** Generate informative plots for analysis and presentation.

## Applications:

`picmol` is a valuable tool for researchers and engineers working in fields such as:

* Chemical engineering
* Materials science
* Pharmaceutical development
* Polymer science

## Getting Started:

Creating an anaconda environment with dependencies.

```python
conda env create -f environment.yml
```

```python
pip install picmol
```

## Examples:

`picmol` includes support for a command line interface, here is an example of displaying the argument options.

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
import pandas as pd
from picmol import Tc_search, load_molecular_properties

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
import pandas as pd
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

# mol_name: molecule name
# mol_id: molecular id in simulations
# density: liquid density at 298.15 K
# mol_class: "extractant", "modifier", "solute", or "solvent"
# smiles: SMILES string for UNIFAC modeling

add_molecule(mol_name, mol_id, density, mol_class, smiles)
```








