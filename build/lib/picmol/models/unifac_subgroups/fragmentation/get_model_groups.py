"""gets groups from a FragmentationModel"""

from typing import Union, List
import pandas as pd
from rdkit import Chem

from .fragmentation_models.fragmentation_model import *
from .get_rdkit_object import *
from .problematics import *


def group_matches(mol_object: Chem.rdchem.Mol, group: str, model: FragmentationModel, action: str="detection") -> tuple:
	"""obtain the group matches in mol_object
	
	Given a functional group (group), a FragmentationModel (subgroups) and a
    RDKit Mol object (mol_object), obtains the SubstructMatches in chem_object
    returns a tuple of tuples containing the atoms that participate in the
    "group" substructure (return of the RDKit GetSubstructMatches function).
    The length of the return tuple is equal to the number of matches of the
    group in the mol_object."""

	if action == "detection":
		mols = model.detection_mols[group]
	elif action == "fit":
		model.fit_mols[group]
	else:
		raise ValueError(f"{action} not valid, use 'detection' or 'fit'")
	
	for mol in mols:
		matches = mol_object.GetSubstructMatches(mol)
		if len(matches) != 0:
			return matches
		
	return matches



def detect_groups(mol_object: Chem.rdchem.Mol, model: FragmentationModel) -> tuple[List[str], List[int]]:
	"""Detect present functional groups in the mol_object molecule
	
	Asks for each functional group in the subgroups DataFrame of the
    FragmentationModel using the SMARTS representation of the functional group.
    Then, returns the detected groups and the number of occurrences."""

	groups = []
	occurrences = []

	for group in model.subgroups.index:
		matches = group_matches(mol_object, group, model)
		how_many_matches = len(matches)
		if how_many_matches > 0:
			groups += [group]
			occurrences += [int(how_many_matches)]
	
	return groups, occurrences



def get_groups(model: FragmentationModel, 
			   identifier: Union[str, Chem.rdchem.Mol], 
			   identifier_type: str = "smiles",) -> dict:
	
	"""obtain FragmentationModel's subgroups of an RDkit Mol object"""

	# RDkit mol object
	mol_object = instantiate_mol_object(identifier, identifier_type)

	# =========================================================================
	# Direct detection of groups presence and occurences
	# =========================================================================
	groups, groups_ocurrences = detect_groups(mol_object=mol_object, model=model)

	# =========================================================================
	# Filter the contribution matrix and sum over row to cancel the contribs
	# =========================================================================
	mol_subgroups = {}
	for g, group in enumerate(groups):
		mol_subgroups[group] = groups_ocurrences[g]

	# =========================================================================
	# Check for the presence of problematic structures and correct.
	# =========================================================================
	mol_subgroups_corrected = correct_problematics(mol_object=mol_object, mol_subgroups=mol_subgroups,model=model)

	if mol_subgroups_corrected == {}:
		return mol_subgroups
	else:
		return mol_subgroups_corrected