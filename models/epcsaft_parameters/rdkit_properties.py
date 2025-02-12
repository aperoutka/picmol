"""Function to compute molecular properties with rdkit"""

from typing import Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

def num_hb_donor(mol):
	''' get number of hydrogen bonding donors '''
	mol = Chem.MolFromSmiles(mol) if type(mol)==str else mol
	mol = AllChem.AddHs(mol)
	count = 0
	for atom in mol.GetAtoms():
		if atom.GetAtomicNum() in [7,8,9]:
			for neighbor in atom.GetNeighbors():
				if neighbor.GetAtomicNum() == 1:  # Check for Hydrogen (1)
					count += 1
	return count

def num_hb_acceptor(mol):
	''' get number of hydrogen bonding acceptors '''
	mol = Chem.MolFromSmiles(mol) if type(mol)==str else mol
	mol = AllChem.AddHs(mol)

	n_donor = num_hb_donor(mol)

	count = 0
	for atom in mol.GetAtoms():
		if atom.GetAtomicNum() in [7,8,9]:
			count +=1 

	if n_donor > count:
		count = 0
		for atom in mol.GetAtoms():
			if atom.GetAtomicNum() in [7,8,9]:
				num_bonds = atom.GetDegree()
				if atom.GetAtomicNum() == 7:  # Nitrogen
					valence_electrons = 5
				elif atom.GetAtomicNum() == 8:  # Oxygen
					valence_electrons = 6
				elif atom.GetAtomicNum() == 9:  # Fluorine
					valence_electrons = 7
				num_lone_pairs = (valence_electrons - num_bonds) / 2
				count += int(num_lone_pairs)

	return count

def get_HB_types(n_donor, n_acceptor):
	assoc_groups = []
	assoc_groups.extend(['P'] * n_donor) 
	assoc_groups.extend(['N'] * n_acceptor)
	return list(assoc_groups) # make sure list is returned

def get_rdkit_mol_properties(smiles_list: Union[str, Chem.Mol]):

	properties = {
		"n_hb_acceptors": np.zeros(len(smiles_list)),
		"n_hb_donors": np.zeros(len(smiles_list)),
		"n_hb_sites": np.zeros(len(smiles_list)),
		"hb_types": [],
		"mw": np.zeros(len(smiles_list)),
		"Vm": np.zeros(len(smiles_list)),
		"q": np.zeros(len(smiles_list)),
	}

	for s, smile in enumerate(smiles_list):
		mol = Chem.MolFromSmiles(smile) if type(smile)==str else mol
		properties["n_hb_acceptors"][s] = num_hb_acceptor(mol)
		properties["n_hb_donors"][s] = num_hb_donor(mol)
		properties["n_hb_sites"][s] = properties["n_hb_acceptors"][s] + properties["n_hb_donors"][s]
		properties["hb_types"].append(get_HB_types(n_acceptor=num_hb_acceptor(mol), n_donor=num_hb_donor(mol)))
		properties["mw"][s] = Descriptors.MolWt(mol)
		properties["q"][s] = Chem.GetFormalCharge(mol)
		molh = AllChem.AddHs(mol)
		AllChem.EmbedMolecule(molh)
		try:
			Vm_cm3_mol = AllChem.ComputeMolVolume(molh)
		except:
			AllChem.EmbedMolecule(mol)
			Vm_cm3_mol = AllChem.ComputeMolVolume(mol)
		properties["Vm"][s] = Vm_cm3_mol * 1E24 / 6.022E23
		
		
	return properties
