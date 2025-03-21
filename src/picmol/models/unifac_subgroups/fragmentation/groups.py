from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import math

from .get_model_groups import *
from .get_rdkit_object import *
from .fragmentation_models.models import *


class Groups:
	"""
	Group class
	Stores the solved FragmentationModels subgroups of a molecule
	"""
	def __init__(self, identifier: str, identifier_type: str = "smiles"):
		self.identifier_type = identifier_type.lower()
		self.identifier = identifier
		self.mol_object = instantiate_mol_object(identifier, identifier_type)
		self.molecular_weight = Descriptors.MolWt(self.mol_object)
		self.unifac = self.SubgroupModel(self.identifier, self.identifier_type, unifac)
		self.unifac_IL = self.SubgroupModel(self.identifier, self.identifier_type, unifac_IL)

	def embed_molecule(self):
		mol = Chem.AddHs(self.mol_object)
		AllChem.EmbedMolecule(mol)
		self._embed_molecule = mol
		return mol

	def radii(self):
		try:
			mol_obj = self._embed_molecule
		except:
			mol_obj = self.embed_molecule()
		ptable = Chem.GetPeriodicTable()
		self._radii = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol_obj.GetAtoms()]
		return self._radii
	
	@property
	def vdw_molar_volume(self):
		'''calculates molar volume in [cm^3/mol]'''
		try:
			radii = self._radii
		except:
			radii = self.radii()
		vdw_v = sum([(4/3)*math.pi*r**3 for r in radii]) * (1E-24) * (6.022E23)
		return vdw_v

	@property
	def vdw_surface_area(self):
		'''calculates surface area in [cm^2/mol]'''
		try:
			radii = self._radii
		except:
			radii = self.radii()
		vdw_sa = sum([4*math.pi*r**2 for r in radii]) * (1E-16) * (6.022E23)
		return vdw_sa

	
	class SubgroupModel:
		def __init__(self, identifier, identifier_type, model):
			self.model = model
			self.identifier_type = identifier_type.lower()
			self.identifier = identifier
			self.subgroups = get_groups(self.model, self.identifier, self.identifier_type)
		
		@property
		def to_num(self):
			subgroup_nums = {}
			for group, occurence in self.subgroups.items():
				group_num = self.model.subgroups.loc[group, "subgroup_id"]
				subgroup_nums[group_num] = occurence
			return subgroup_nums
		
		@property
		def r(self):
			mol_r = 0
			for group, occurence in self.subgroups.items():
				subgroup_r = self.model.RQ.loc[group, "R"]
				mol_r += occurence * subgroup_r
			return mol_r

		@property
		def q(self):
			mol_q = 0
			for group, occurence in self.subgroups.items():
				subgroup_q = self.model.RQ.loc[group, "Q"]
				mol_q += occurence * subgroup_q
			return mol_q

		@property
		def subgroup_q(self):
			q_dict = {}
			for group, occurence in self.subgroups.items():
				group_num = self.model.subgroups.loc[group, "subgroup_id"]
				q = self.model.RQ.loc[group, "Q"]
				q_dict[group_num] = q
			return q_dict

