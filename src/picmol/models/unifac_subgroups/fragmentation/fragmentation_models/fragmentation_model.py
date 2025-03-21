import json
from typing import List, Union
import numpy as np
import pandas as pd
from rdkit import Chem


class FragmentationModel:
	
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

		# =====================================================================
		# Empty problematics template
		# =====================================================================
		if problematic_structures is None:
			self.problematic_structures = pd.DataFrame(
				[], columns=["smarts", "contribute"]
			).set_index("smarts")
		else:
			self.problematic_structures = problematic_structures

		# =====================================================================
		# Contribution matrix build
		# =====================================================================
		self.contribution_matrix = self._build_contrib_matrix()

		# =====================================================================
		# Instantiate all de mol object from their smarts
		# =====================================================================
		self.detection_mols = self._instantiate_detection_mol()
		self.fit_mols = self._instantiate_fit_mols()	


	def _build_contrib_matrix(self) -> pd.DataFrame:
		"""build contribution matrix of model"""	

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
		"""Instantiate all the rdkit Mol object from the detection_smarts."""

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
		"""Instantiate all the rdkit Mol object from the smarts."""

		mols = {}

		for group in self.subgroups.index:
			smarts = self.subgroups.loc[group, "smarts"]

			if isinstance(smarts, str):
				mols[group] = [Chem.MolFromSmarts(smarts)]
			else:
				mols[group] = self.detection_mols[group]
		
		return mols
		