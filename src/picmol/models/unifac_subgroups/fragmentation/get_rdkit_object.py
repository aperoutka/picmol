from functools import cache
from typing import Union
from rdkit import Chem


@cache
def instantiate_mol_object(identifier: Union[str, Chem.rdchem.Mol], identifier_type: str = "smiles") -> Chem.rdchem.Mol:
  """
  Instantiate a RDKit Mol object from SMILES or an existing Mol object.

  :param identifier: The molecular identifier, either a SMILES string or an RDKit Mol object.
  :type identifier: Union[str, Chem.rdchem.Mol]
  :param identifier_type: The type of identifier provided.  Must be 'smiles' or 'mol'.
  :type identifier_type: str, optional
  :return: An RDKit Mol object.
  :rtype: Chem.rdchem.Mol
  :raises ValueError: If the identifier type is invalid or if a 'mol' type identifier is not an RDKit Mol object.

  .. note::
      The function is cached, so repeated calls with the same identifier
      will return the same Mol object.
  """

  if identifier_type.lower() == "smiles":
    smiles = identifier
    chem_object = Chem.MolFromSmiles(smiles)

  elif identifier_type.lower() == "mol":
    chem_object = identifier

    if not isinstance(chem_object, Chem.rdchem.Mol):
      raise ValueError(
        "If 'mol' identifier type is used, the identifier must be a "
        "rdkit.Chem.Chem.rdchem.Mol object."
      )

  else:
    raise ValueError(
      f"Identifier type: {identifier_type} not valid, use: "
      "'smiles' or 'mol'"
    )

  return chem_object
