

def main():

  print('''
PICMOL: A Phase Instability Calculator for Molecular Design
        
objective: display key features of the package along with argument parsing



add_molecule(mol_name, mol_id, mol_class, smiles, density=None)
description: add a molecule to the molecular properties database
arguments:        
  - mol_name: name of the molecule (e.g. "hexane")
  - mol_id: id of the molecule in .top file (e.g. "HEXAN")
  - mol_class: class of the molecule (e.g. "extractant", "solute", "solvent")
  - smiles: SMILES representation of the molecule (e.g. "CCCCCC")
  - density: density of the molecule (e.g. 0.661 g/mL)        

           
load_molecular_properties(index)
description: load the molecular properties database
arguments:
  - index: index of the dataframe (e.g. "mol_name", "mol_id")        



KBI(
    prj_path, 
    pure_component_path, 
    rdf_dir, kbi_method, 
    thermo_limit_extrapolate, 
    rkbi_min,
    kbi_fig_dirname,
    avg_start_time,
    avg_end_time,
    solute_mol,
    geom_mean_pairs
  )       
description: calculate the KBI (Kirkwood-Buff Integral) for a given project
arguments:
  - prj_path: path to the project directory (e.g. "/path/to/project")
  - pure_component_path: path to the pure component directory (e.g. "/path/to/pure_component")
  - rdf_dir: RDF directory name (e.g. "rdf_files")
  - kbi_method: method for KBI calculation (e.g. "adj", "kgv")
  - thermo_limit_extrapolate: method for thermodynamic limit extrapolation (e.g. True, False)
  - rkbi_min: fraction of r corresponding to minimum RKBI value (e.g. 0.75)
  - kbi_fig_dirname: directory name for KBI figures (e.g. "kbi_analysis")
  - avg_start_time: start time (ns) for analysis (e.g. 100)
  - avg_end_time: end time (ns) for analysis (e.g. end of trajectory)
  - solute_mol: solute molecule id (e.g. "HEXAN")
  - geom_mean_pairs: geometric mean pairs (list of lists, e.g. [["DMDBP","HEXAN"]])

        

ThermoModel(
    model_name,
    KBIModel,
    which_molar_vol,
    unif_version,
    identifiers,
    identifier_type,
    Tmin,
    Tmax,
    dT
    )
description: create a thermodynamic model for temperature scaling & LLE calculations
arguments:
  - model_name: name of the model (e.g. "quartic", "unifac", "uniquac", "nrtl", "fh")
  - KBIModel: KBI object (e.g. kbi_obj)
  - which_molar_vol: how to calculate molar vol (if using KBI model, e.g. "sim" or "md" vs. "exp")
  - Tmin: minimum temperature (e.g. 100 K)
  - Tmax: maximum temperature (e.g. 400 K)
  - dT: temperature step (e.g. 1 K)
        
  ~ ONLY IF KBI WAS NOT RUN, E.G. UNIFAC WITHOUT KBI. ~
    - unif_version: version of UNIFAC model (e.g. "unifac")   
    - identifiers: list of identifiers if (e.g. ["DMDBP", "HEXAN"])
    - identifier_type: type of identifiers (e.g. "mol_id", "mol_name")
        ''')