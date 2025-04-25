import numpy as np


  
def get_solute_molid(mols, mol_class):
  '''get solute molid from the list of molecules, using priority of mol_class'''
  solute_types = ["solute", "extractant", "modifier", "solvent"]
  for i, mol in enumerate(mols):
    for j, solute_type in enumerate(solute_types):
      if mol_class[mol].lower() == solute_type.lower():
        return mol

