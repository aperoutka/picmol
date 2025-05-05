from typing import Union, List
import pandas as pd
from rdkit import Chem

from .fragmentation_models.fragmentation_model import *
from .get_rdkit_object import *
from .problematics import *


def group_matches(mol_object: Chem.rdchem.Mol, group: str, model: FragmentationModel, action: str="detection") -> tuple:
  r"""
  Obtain the group matches in a molecule.

  Given a functional group, a FragmentationModel, and an RDKit Mol object,
  this function obtains the SubstructMatches in the molecule.

  :param mol_object: The RDKit Mol object in which to search for the group.
  :type mol_object: Chem.rdchem.Mol
  :param group: The functional group to search for.
  :type group: str
  :param model: The FragmentationModel containing the group definitions.
  :type model: FragmentationModel
  :param action: The action context ('detection' or 'fit').
  :type action: str, optional
  :return: A tuple of tuples containing the atoms that participate in the
            group substructure.  The length of the tuple equals the number of
            matches of the group in the molecule.
  :rtype: tuple
  :raises ValueError: If the action is not 'detection' or 'fit'.
  """

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
  r"""
  Detect functional groups in a molecule.

  This function identifies the functional groups present in a molecule,
  using the SMARTS representations defined in a FragmentationModel.

  :param mol_object: The RDKit Mol object to analyze.
  :type mol_object: Chem.rdchem.Mol
  :param model: The FragmentationModel containing the subgroup definitions.
  :type model: FragmentationModel
  :return: A tuple containing a list of detected group names and a list of
            their corresponding occurrences.
  :rtype: tuple[List[str], List[int]]
  """

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
  
  r"""
  Obtain subgroups from an RDKit Mol object.

  This function retrieves the subgroups of an RDKit Mol object based on the
  definitions in a FragmentationModel.  It also handles problematic
  structures and returns corrected subgroup counts.

  :param model: The FragmentationModel containing the subgroup definitions.
  :type model: FragmentationModel
  :param identifier: The molecular identifier, either a SMILES string or an
                      RDKit Mol object.
  :type identifier: Union[str, Chem.rdchem.Mol]
  :param identifier_type: The type of identifier provided ('smiles' or 'mol').
  :type identifier_type: str, optional
  :return: A dictionary of subgroups and their occurrences in the molecule.
  :rtype: dict
  """

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