import numpy as np

def mol2vol(val, V0):
	'''convert mol frac to vol frac'''
	# check that datatypes are arrays
	V0 = np.array(V0)
	if type(val) == float:
		z_arr = np.array([val, 1-val])
	else:
		z_arr = np.array(val)
	vbar = z_arr @ V0
	try:
		v_arr = z_arr * V0 / vbar[:,np.newaxis]
		return v_arr
	except:
		v_arr = z_arr * V0 / vbar
		return v_arr[0]

  
def get_solute_molid(mols, mol_class):
  '''get solute molid from the list of molecules, using priority of mol_class'''
  solute_types = ["solute", "extractant", "modifier", "solvent"]
  for i, mol in enumerate(mols):
    for j, solute_type in enumerate(solute_types):
      if mol_class[mol].lower() == solute_type.lower():
        return mol

