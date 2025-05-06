picmol: Phase Instability Calculator for MOLecular design
==========================================================

.. toctree::
  :maxdepth: 4
  :caption: API Reference:
  :titlesonly:

  picmol.get_molecular_properties
  picmol.kbi
  picmol.thermo_model
  picmol.models
  picmol.plotter

.. toctree::
  :maxdepth: 2
  :caption: Examples:

  examples/01_kbi_analysis.ipynb
  examples/02_thermo_models.ipynb
  examples/03_combined_kbi_thermo_analysis.ipynb
  examples/04_unifac_lle.ipynb
  examples/05_unifac_Tc_search.ipynb


Installation
-------------
.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
  :target: https://picmol.readthedocs.io/
  :alt: docs
.. image:: http://img.shields.io/badge/License-MIT-blue.svg
  :target: https://tldrlegal.com/license/mit-license
  :alt: license
.. image:: https://img.shields.io/badge/Python-3.10%2B-blue

``picmol`` can be installed from cloning github repository.

.. code-block:: text

  git clone https://github.com/aperoutka/picmol.git

Creating an anaconda environment with ``picmol`` dependencies and install ``picmol``.

.. code-block:: text
  
  cd picmol
  conda env create -f environment.yml
  conda activate picmol
  pip install .


File Organization
------------------

.. code-block:: text
  :caption: Project File Structure

  project/
  └── system/
      ├── rdf_files/
      │   ├── mol1_mol1.xvg
      │   ├── mol1_mol2.xvg
      │   └── mol1_mol2.xvg
      ├── system_npt.edr
      └── system.top

.. code-block:: text
  :caption: Pure Component File Structure

  pure_components/
  └── molecule_temp/
      ├── molecule_temp_npt.edr
      └── molecule_temp.top

Indices and tables
===================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`