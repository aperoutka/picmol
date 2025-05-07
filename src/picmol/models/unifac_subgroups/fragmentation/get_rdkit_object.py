from functools import cache
from typing import Union
from rdkit import Chem


@cache
def instantiate_mol_object(identifier: Union[str, Chem.rdchem.Mol]) -> Chem.rdchem.Mol:
  """
  Instantiate a RDKit Mol object from SMILES or an existing Mol object.

  :param identifier: The molecular identifier, either a SMILES string or an RDKit Mol object.
  :type identifier: str or Chem.rdchem.Mol
  :return: An RDKit Mol object.
  :rtype: Chem.rdchem.Mol
  :raises ValueError: If the identifier type is invalid or if a  `mol` type identifier is not an RDKit Mol object.

  .. note::
      The function is cached, so repeated calls with the same identifier
      will return the same Mol object.
  """

  try:
    chem_object = Chem.MolFromSmiles(identifier)
  except:
    chem_object = identifier
    if not isinstance(chem_object, Chem.rdchem.Mol):
      raise ValueError(
        "If 'mol' identifier type is used, the identifier must be a "
        "rdkit.Chem.Chem.rdchem.Mol object."
      )

  return chem_object
