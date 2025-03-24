from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
import pandas as pd
from pathlib import Path
import sys, os

def add_molecule(mol_name: str, mol_id: str, density: float, mol_class: str, smiles: str):
	'''mol_class: solute or solvent'''
	mol_obj = Chem.MolFromSmiles(smiles)
	mol_obj = AllChem.AddHs(mol_obj)

	def get_electron_number(mol):
		atomic_numbers = []
		for atom in mol.GetAtoms():
			atomic_numbers += [atom.GetAtomicNum()]
		return sum(atomic_numbers)

	n_electrons = get_electron_number(mol_obj)
	mol_wt = Chem.Descriptors.MolWt(mol_obj)
	molarity = 1000*density/mol_wt
	molar_vol = mol_wt/density

	# get solvent accessible surface area (SASA)
	AllChem.EmbedMolecule(mol_obj, useRandomCoords=True)
	radii = rdFreeSASA.classifyAtoms(mol_obj)
	sasa = rdFreeSASA.CalcSASA(mol_obj, radii=radii)

	new_molecule = {
		'mol_name': mol_name.upper(),
		'cosmo_name': mol_name.lower(),
		'mol_id': mol_id.upper(),
		'mol_wt': f"{mol_wt:.3f}",
		'density': f"{density:.3f}",
		'molarity': f"{molarity:.3f}",
		'molar_vol': f"{molar_vol:.3f}",
		'sasa': f"{sasa:.3f}",
		'n_electrons': f"{n_electrons:.0f}",
		'mol_class': mol_class,
		'smiles': smiles,
	}

	# read in dataframe and add molecule and write back out
	df = pd.read_csv(Path(__file__).parent / "data" / "molecular_properties.csv")
	df = df._append(new_molecule, ignore_index=True)
	df.to_csv(Path(__file__).parent / "data" / "molecular_properties.csv", index=False)

	
def search_molecule(mol, index):
	'''takes in molecule name search csv to find molecule'''
	df = load_molecular_properties(index)
	if mol not in df.index:
		print('molecule not found!!!')
		print('use "add_molecule()" method to add to molecule list')
		sys.exit()


def load_molecular_properties(index):
	return pd.read_csv(Path(__file__).parent / "data" / "molecular_properties.csv").set_index(index)	
	