from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
import pandas as pd
from pathlib import Path
import sys, os

def add_molecule(mol_name: str, mol_id: str, mol_class: str, smiles: str, density: float = None):
	"""
	Adds a molecule to `molecular_properties.csv` file, using user specified information and rdkit properties
	
	:param mol_name: molecule name used in figure labels
	:param mol_id: molecule name used in .top files 
	:param mol_class: str to describe molecule type (i.e., 'solute', 'extractant', 'solvent')
	:param smiles: SMILES str to identify molecule using RDkit
	:param density: mass density (g/mL) of molecule (optional), if not provided density and molar volume are estimated with RDkit
	"""
	mol_obj = Chem.MolFromSmiles(smiles)
	mol_obj = AllChem.AddHs(mol_obj)

	def get_electron_number(mol):
		atomic_numbers = []
		for atom in mol.GetAtoms():
			atomic_numbers += [atom.GetAtomicNum()]
		return sum(atomic_numbers)

	n_electrons = get_electron_number(mol_obj)
	mol_wt = Chem.Descriptors.MolWt(mol_obj)
	mol_charge = Chem.GetFormalCharge(mol_obj)
	
	if density is not None:
		molar_vol = mol_wt/density
	else:
		AllChem.EmbedMolecule(mol_obj, useRandomCoords=True)
		molar_vol = AllChem.ComputeMolVolume(mol_obj)
		density = mol_wt/molar_vol 

	molarity = 1000*density/mol_wt

	new_molecule = {
		'mol_name': mol_name.upper(),
		'mol_id': mol_id.upper(),
		'mol_wt': f"{mol_wt:.3f}",
		'density': f"{density:.3f}",
		'molarity': f"{molarity:.3f}",
		'molar_vol': f"{molar_vol:.3f}",
		'n_electrons': f"{n_electrons:.0f}",
		'mol_charge': f"{mol_charge:.0f}",
		'mol_class': mol_class,
		'smiles': smiles,
	}

	# read in dataframe and add molecule and write back out
	df = pd.read_csv(Path(__file__).parent / "data" / "molecular_properties.csv")
	df = df._append(new_molecule, ignore_index=True)
	df.to_csv(Path(__file__).parent / "data" / "molecular_properties.csv", index=False)

	
def search_molecule(mol, index):
	"""
	Uses molecule name to search `molecular_properties.csv` to determine if molecule is present

	:param mol: molecule to look up of type index
	:param index: type of molecule id (i.e., 'mol_name', 'mol_id', 'smiles')
	:raises SystemExit: If the molecule is not found.
	"""
	df = load_molecular_properties(index)
	if mol not in df.index:
		print('molecule not found!!!')
		print('use "add_molecule()" method to add to molecule list')
		sys.exit()


def load_molecular_properties(index):
	"""
	Loads `molecular_properties.csv` with pandas setting index to specified column

	:param index: column to set as index in pandas DataFrame (options: 'mol_name', 'mol_id', 'smiles')
	:return: pandas DataFrame of `molecular_properties.csv` with specified index
	"""
	return pd.read_csv(Path(__file__).parent / "data" / "molecular_properties.csv").set_index(index)	
	