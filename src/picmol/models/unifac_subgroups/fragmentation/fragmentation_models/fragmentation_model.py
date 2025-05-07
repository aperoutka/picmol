import json
from typing import List, Union
import numpy as np
import pandas as pd
from rdkit import Chem


class FragmentationModel:
  """
  A model for fragmenting molecules and analyzing their subgroups.

  :param subgroups:
      A pandas DataFrame defining the molecular subgroups.
      The index of the DataFrame represents the subgroup IDs.
      It must contain columns named "smarts" and "contribute".
  :type subgroups: pd.DataFrame
  :param RQ:
      A pandas DataFrame representing the subgroup R & Q UNIFAC terms.
  :type RQ: pd.DataFrame
  :param split_detection_smarts:
      A list of subgroup identifiers (corresponding to the index in the
      `subgroups` DataFrame) for which the SMARTS patterns for detection
      are split into multiple patterns.
  :type split_detection_smarts: list[str], optional
  :param problematic_structures:
      A pandas DataFrame describing problematic structural patterns.
      The index of the DataFrame contains SMARTS patterns, and the
      DataFrame must contain a column named "contribute".
      The "contribute" column should contain string representations of
      dictionaries. These dictionaries map subgroup identifiers to the
      contribution factors for each problematic structure.
      If None, an empty DataFrame is created.
  :type problematic_structures: pd.DataFrame or None, optional

  :ivar contribution_matrix:
      A pandas DataFrame representing the contribution matrix,
      derived from the "contribute" column of the `subgroups` DataFrame.
  :vartype contribution_matrix: pd.DataFrame
  :ivar detection_mols:
      A dictionary mapping subgroup IDs to lists of RDKit Mol
      objects, instantiated from the SMARTS patterns in the "smarts"
      column of the ``subgroups`` DataFrame.  For subgroups in
      ``split_detection_smarts``, the value is a list of multiple Mol objects.
  :vartype detection_mols: dict
  :ivar fit_mols:
      A dictionary mapping subgroup IDs to lists of RDKit Mol
      objects, similar to ``detection_mols``, but using the original
      SMARTS patterns from the "smarts" column of the ``subgroups`` DataFrame,
      even for subgroups in ``split_detection_smarts``.
  :vartype fit_mols: dict[int, list[rdkit.Chem.rdchem.Mol]]
  """
  
  def __init__(
      self, 
      subgroups: pd.DataFrame,
      RQ: pd.DataFrame,
      split_detection_smarts: List[str] = [],
      problematic_structures: Union[pd.DataFrame, None] = None,
    ) -> None:
    self.subgroups = subgroups
    self.split_detection_smarts = split_detection_smarts
    self.RQ = RQ

    # Empty problematics template
    if problematic_structures is None:
      self.problematic_structures = pd.DataFrame(
        [], columns=["smarts", "contribute"]
      ).set_index("smarts")
    else:
      self.problematic_structures = problematic_structures

    # Contribution matrix build
    self.contribution_matrix = self._build_contrib_matrix()

    # Instantiate all de mol object from their smarts
    self.detection_mols = self._instantiate_detection_mol()
    self.fit_mols = self._instantiate_fit_mols()  


  def _build_contrib_matrix(self) -> pd.DataFrame:
    """
    Builds the contribution matrix for the model. The matrix is derived
    from the "contribute" column of the `subgroups` DataFrame, which
    specifies how each subgroup contributes to the count of other subgroups.

    :returns: A pandas DataFrame representing the contribution matrix.
    :rtype: pd.DataFrame
    """  

    index = self.subgroups.index.to_numpy()
    matrix = np.zeros((len(index), len(index)), dtype=int)


    # build matrix
    dfm = pd.DataFrame(matrix, index=index, columns=index).rename_axis("subgroup")

    # fill matrix
    for group in index:
      str_contribution = self.subgroups.loc[group, "contribute"]

      try:
        contribution = json.loads(str_contribution)
      except json.JSONDecodeError:
        raise ValueError(f"Bad contribute parsing of group: {group}")
      except TypeError:
        raise TypeError(f"Bad contribute parsing of group: {group}")
    
    for k in contribution.keys():
      dfm.loc[group, k] = contribution[k]

    return dfm
  

  def _instantiate_detection_mol(self) -> dict:
    """
    Creates RDKit Mol objects from the SMARTS patterns associated with each
    subgroup in the `subgroups` DataFrame.  For subgroups whose SMARTS
    patterns are split (as indicated in `self.split_detection_smarts`),
    multiple Mol objects are created.

    :returns: A dictionary mapping subgroup identifiers to lists of RDKit Mol objects.
    :rtype: dict
    """


    mols = {}

    for group in self.subgroups.index:
      if group not in self.split_detection_smarts:
        mols[group] = [Chem.MolFromSmarts(self.subgroups.loc[group, "smarts"])]
      else:
        smarts = self.subgroups.loc[group, "smarts"].split(",")
        mol_smarts = []
        for sms in smarts:
          mol_smarts += [Chem.MolFromSmarts(sms)]
        
        mols[group] = mol_smarts
    
    return mols



  def _instantiate_fit_mols(self) -> dict:
    """
    Creates RDKit Mol objects from the SMARTS patterns associated with each
    subgroup in the `subgroups` DataFrame. This function differs from
    `_instantiate_detection_mol` in that it uses the original SMARTS patterns
    from the "smarts" column, even for subgroups whose SMARTS patterns are
    split for detection purposes.

    :returns: A dictionary mapping subgroup identifiers to lists of RDKit Mol objects.
    :rtype: dict
    """

    mols = {}

    for group in self.subgroups.index:
      smarts = self.subgroups.loc[group, "smarts"]

      if isinstance(smarts, str):
        mols[group] = [Chem.MolFromSmarts(smarts)]
      else:
        mols[group] = self.detection_mols[group]
    
    return mols
    