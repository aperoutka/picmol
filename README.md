# picmol: Molecular Simulation Analysis and Phase Equilibria Calculations

`picmol` is a `Python` library designed to streamline the analysis of molecular dynamics simulations, focusing on the application of Kirkwood-Buff (KB) theory for the calculation of activity coefficients, excess thermodynamic properties, and liquid-liquid equilibria (LLE).

.. image:: http://img.shields.io/pypi/v/picmol.svg?style=flat
  :target: https://pypi.python.org/pypi/picmol
  :alt: Version_status
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
  :target: https://github.com/aperoutka/picmol/blob/main/LICENSE
  :alt: license
.. image:: https://img.shields.io/pypi/pyversions/picmol.svg
  :target: https://pypi.python.org/pypi/picmol
  :alt: Supported_versions

**Key Features:**

* **Kirkwood-Buff Integration:**
    * Calculates interaction parameters from molecular dynamics simulations using Kirkwood-Buff theory.
    * Fits these parameters to various thermodynamic models, including:
        * Numerical models (e.g., quartic)
        * UNIQUAC
        * NRTL
        * Flory-Huggins
* **Liquid-Liquid Equilibria (LLE) Calculations:**
    * Performs phase diagram calculations for LLE systems, leveraging the KB-derived interaction parameters.
    * Generates phase diagrams for visual analysis.
* **Small-Angle X-ray Scattering (SAXS) Support:**
    * Includes functionality to calculate Io from SAXS data, providing valuable structural information.
* **Visualization Tools:**
    * Offers comprehensive visualization capabilities for both Kirkwood-Buff analysis results and LLE phase diagrams, aiding in data interpretation.

**Workflow:**

`picmol` provides a cohesive workflow that bridges the gap between molecular dynamics simulations and phase equilibria calculations. It enables users to:

1.  **Analyze MD simulations:** Use Kirkwood-Buff theory to derive interaction parameters.
2.  **Fit thermodynamic models:** Fit the derived parameters to suitable thermodynamic models.
3.  **Calculate LLE:** Predict liquid-liquid equilibria and generate phase diagrams.
4.  **Analyze SAXS data:** Derive Io values from SAXS data.
5.  **Visualize results:** Generate informative plots for analysis and presentation.

**Applications:**

`picmol` is a valuable tool for researchers and engineers working in fields such as:

* Chemical engineering
* Materials science
* Pharmaceutical development
* Polymer science

**Getting Started:**

Creating an anaconda environment with dependencies.

```python
conda env create -f environment.yml
```

pip install picmol


