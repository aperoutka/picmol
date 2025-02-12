""" functions to compute morgan fingerprints for molecules """

from typing import Callable, List, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray 


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048

def compute_morgan_fingerprints(
			mols: List[str], 
			radius: int = MORGAN_RADIUS,
			num_bits: int = MORGAN_NUM_BITS,
) -> Union[np.ndarray, None]:

		""" generates a binary Morgan fingerprint for each molecule """

		fingerprints = []
		for m, mol in enumerate(mols):
			try:
				mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol 
				morgan_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
				morgan_fp = np.zeros((1,))
				ConvertToNumpyArray(morgan_vec, morgan_fp)
				fingerprints.append(morgan_fp.astype(bool))
			except Exception as e:
				fingerprints.append(np.zeros((num_bits), dtype=bool))
		
		return np.array(fingerprints)
