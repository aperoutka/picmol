from rdkit import Chem
from rdkit.Chem import AllChem
import json 
from pathlib import Path

class FragmentMolecule:

	def __init__(self, smiles):
		self.smiles = smiles
		self.mol = Chem.MolFromSmiles(self.smiles)
		self.n_heavy_atoms = self.mol.GetNumHeavyAtoms()

		with open(Path(__file__).parent / "subgroup_smarts.json", "r") as f:
			self.smarts = json.load(f)


	@property
	def groups(self):
		"""perform fragmentation"""
		fragments, _ = self.adjust_problematic_fragments()
		return {group: fragments.count(group) for group in set(fragments)}


	def fragment_molecule(self):
		# find the location of all fragment using the given smarts
		matches = dict((g, self.mol.GetSubstructMatches(Chem.MolFromSmarts(self.smarts[g]))) for g in self.smarts)
		bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in self.mol.GetBonds()]
		return self.convert_matches(matches, bonds)


	def convert_matches(self, matches, bonds):
		# check if every atom is captured by exatly one fragment
		identified_atoms = [i for l in matches.values() for k in l for i in k]
		unique_atoms = set(identified_atoms)
		# print(self.n_heavy_atoms, len(unique_atoms), len(identified_atoms), '\n', matches)
		if len(unique_atoms) == len(identified_atoms) and len(unique_atoms) == self.n_heavy_atoms:
			# Translate the atom indices to segment indices (some segments contain more than one atom)
			segment_indices = sorted((sorted(k), group) for group, l in matches.items() for k in l)
			segments = [group for _, group in segment_indices]
			segment_map = [i for i, (k, _) in enumerate(segment_indices) for j in k]
			bonds = [(segment_map[a], segment_map[b]) for a, b in bonds]
			bonds = [(a, b) for a, b in bonds if a != b]
			return segments, bonds
		raise Exception("Molecule cannot be fragmented with the given SMARTS!")

		
	def adjust_problematic_fragments(self):
		segments, bonds = self.fragment_molecule()
		self.mol = AllChem.AddHs(self.mol)
		AllChem.EmbedMolecule(self.mol)
		Chem.GetSymmSSSR(self.mol)
		rings = self.mol.GetRingInfo().NumRings()
		
		if rings > 1:
			raise Exception("Only molecules with up to 1 ring are allowed!")
				
		# segments, bonds = self.convert_ether_groups(segments, bonds)
		# segments, bonds = self.convert_alkyne_groups(segments, bonds)
		return segments, bonds


	def convert_ether_groups(self, segments, bonds):
		ethers = sum(1 for group in segments if group == "O")
		if ethers == 0:
			return segments, bonds
		elif ethers == 1:
			index = sum(i for i, group in enumerate(segments) if group == "O")
			for a, b in bonds:
				if a == index:
					right = (b, segments[b])
				elif b == index:
					left = (a, segments[a])

			if left[1] == "CH3":
				index2 = left[0]
				new_group = "OCH3"
			elif right[1] == "CH3":
				index2 = right[0]
				new_group = "OCH3"
			elif left[1] == "CH2":
				index2 = left[0]
				new_group = "OCH2"
			elif right[1] == "CH2":
				index2 = right[0]
				new_group = "OCH2"
			else:
				raise Exception("The ether is not compatible with the groups by Sauer et al.!")
			
			matches = {new_group: ((index, index2),)}
			for i, group in enumerate(segments):
				if i in {index, index2}:
					continue
				if group in matches:
					matches[group] = matches[group] + ((i,),)
				else:
					matches[group] = ((i,),)
			return self.convert_matches(matches, bonds)
		
		else:
			raise Exception("The conversion is only implemented for molecules with one ether group!")


	def convert_alkyne_groups(self, segments, bonds):
		matches = {'CtCH': ()}
		for i,group in enumerate(segments):
			if group not in {"tC", "tCH"}:
				if group in matches:
					matches[group] += ((i,),)
				else:
					matches[group] = ((i,),)
		for i,j in bonds:
			if segments[i] == "tC" and segments[j] == "tC":
				raise Exception("Only terminal alkynyl groups are allowed!")
			elif segments[i] == "tC" and segments[j] == "tCH":
				matches['CtCH'] += ((i,j),)
			elif segments[i] == "tCH" and segments[j] == "tC":
				matches['CtCH'] += ((j,i),)
		return self.convert_matches(matches, bonds)
