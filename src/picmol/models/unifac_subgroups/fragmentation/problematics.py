import json
from rdkit import Chem
from .fragmentation_models.fragmentation_model import *


def correct_problematics(
  mol_object: Chem.rdchem.Mol,
  mol_subgroups: dict,
  model: FragmentationModel,
) -> dict:
  
  """
  Correct problematic structures in mol_object.

  Identifies and corrects problematic structural patterns within a molecule
  by adjusting the counts of its subgroups. It uses a fragmentation model to
  determine which substructures are problematic and how their contributions
  should be redistributed.

  :param mol_object:
      The molecule to be corrected, represented as an RDKit Mol object.
  :type mol_object: Chem.rdchem.Mol
  :param mol_subgroups:
      A dictionary containing the initial counts of subgroups in the molecule.
      The keys are subgroup identifiers (e.g., SMARTS strings), and the values
      are their corresponding counts.
  :type mol_subgroups: dict
  :param model:
      A FragmentationModel object that provides information about problematic
      structures and their correction factors.  It must have an attribute
      ``problematic_structures`` which is a pandas DataFrame.  The index of the
      DataFrame contains SMARTS patterns, and the DataFrame must contain a
      column named "contribute".  The "contribute" column should contain
      string representations of dictionaries.  These dictionaries map subgroup
      identifiers to the contribution factors for each problematic structure.
  :type model: FragmentationModel

  :returns: A dictionary representing the corrected subgroup counts.  The keys are
      subgroup identifiers, and the values are their corrected counts.
      Subgroups with a final count of 0 are removed from the dictionary.
  :rtype: dict
  """

  corrected_subgroups = mol_subgroups.copy()

  for smarts in model.problematic_structures.index:
    structure = Chem.MolFromSmarts(smarts)
    matches = mol_object.GetSubstructMatches(structure)
    how_many_problems = len(matches)

    if how_many_problems > 0:
      problematic_dict = json.loads(model.problematic_structures.loc[smarts, "contribute"])

      for subgroup, contribution in problematic_dict.items():
        corrected_subgroups[subgroup] = (corrected_subgroups.get(subgroup, 0) + contribution * how_many_problems)

  # Eliminate occurrences == 0
  corrected_subgroups = {key: value for key, value in corrected_subgroups.items() if value != 0}
  
  return corrected_subgroups
