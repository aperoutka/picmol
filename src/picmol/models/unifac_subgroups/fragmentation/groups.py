from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import math

from .get_model_groups import *
from .get_rdkit_object import *
from .fragmentation_models.models import *


class Groups:
  """
  Group class. Stores the solved FragmentationModels subgroups of a molecule.

  :param identifier: The molecular identifier (SMILES string or RDKit Mol object).
  :type identifier: str or rdkit.rdchem.Mol

  :ivar mol_object: The RDKit Mol object representing the molecule.
  :vartype mol_object: Chem.rdchem.Mol
  :ivar molecular_weight: The molecular weight of the molecule.
  :vartype molecular_weight: float
  :ivar unifac: The SubgroupModel object for the UNIFAC model.
  :vartype unifac: Groups.SubgroupModel
  :ivar unifac_IL: The SubgroupModel object for the UNIFAC-IL model.
  :vartype unifac_IL: Groups.SubgroupModel
  """
  def __init__(self, identifier):
    self.identifier = identifier
    self.mol_object = instantiate_mol_object(identifier)
    self.molecular_weight = Descriptors.MolWt(self.mol_object)
    self.unifac = self.SubgroupModel(self.identifier, unifac)
    self.unifac_IL = self.SubgroupModel(self.identifier, unifac_IL)

  def embed_molecule(self):
    mol = Chem.AddHs(self.mol_object)
    AllChem.EmbedMolecule(mol)
    self._embed_molecule = mol
    return mol

  def radii(self):
    r"""
    Calculates the van der Waals radii for each atom in the molecule using
    RDKit's GetPeriodicTable.

    :return: A list of van der Waals radii for each atom.
    :rtype: List[float]
    """
    try:
      mol_obj = self._embed_molecule
    except:
      mol_obj = self.embed_molecule()
    ptable = Chem.GetPeriodicTable()
    self._radii = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol_obj.GetAtoms()]
    return self._radii
  
  @property
  def vdw_molar_volume(self):
    R"""
    Calculates the van der Waals molar volume in cm\ :sup:`3`/mol based on the
    atomic radii.

    .. math::
        V_{vdw} = N_A \sum_i \frac{4}{3} \pi r_i^3

    :return: The van der Waals molar volume in cm\ :sup:`3`/mol.
    :rtype: float
    """
    try:
      radii = self._radii
    except:
      radii = self.radii()
    vdw_v = sum([(4/3)*math.pi*r**3 for r in radii]) * (1E-24) * (6.022E23)
    return vdw_v

  @property
  def vdw_surface_area(self):
    r"""
    Calculates the van der Waals surface area in cm\ :sup:`2`/mol based on the
    atomic radii.

    .. math::
        SA_{vdw} = N_A \sum_i 4 \pi r_i^2

    :return: The van der Waals surface area in cm\ :sup:`2`/mol.
    :rtype: float
    """
    try:
      radii = self._radii
    except:
      radii = self.radii()
    vdw_sa = sum([4*math.pi*r**2 for r in radii]) * (1E-16) * (6.022E23)
    return vdw_sa

  
  class SubgroupModel:
    """
    Handles subgroup-related calculations for a specific UNIFAC model.

    :param identifier:
        The identifier of the molecule. This could be a SMILES string or
        rdkit.rdchem.Mol object.
    :type identifier: str or rdkit.rdchem.Mol
    :param model: The fragmentation model instance containing the subgroup definitions and parameters.
    :type model: FragmentationModel

    :ivar subgroups: A dictionary mapping subgroup names to their occurrences in the molecule.
    :vartype subgroups: dict
    """
    def __init__(self, identifier, model):
      self.model = model
      self.identifier = identifier
      self.subgroups = get_groups(self.model, self.identifier)
    
    @property
    def to_num(self):
      r"""
      Converts subgroup names to subgroup IDs. Creates a dictionary mapping subgroup IDs to their occurrences in the molecule.

      :returns: A dictionary of subgroup IDs and their occurrences.
      :rtype: dict
      """
      subgroup_nums = {}
      for group, occurence in self.subgroups.items():
        group_num = self.model.subgroups.loc[group, "subgroup_id"]
        subgroup_nums[group_num] = occurence
      return subgroup_nums
    
    @property
    def r(self):
      r"""
      Calculates the molar van der Waals volume parameter (:math:`r_i`) for molecule :math:`i`, from
      the summation of volume parameters (:math:`R_j`) for subgroup :math:`j` multiplied by
      their occurrences.

      .. math::
          r_i = \sum_j \nu_{ji} R_j

      :return: molar van der Waals volume parameter.
      :rtype: float
      """
      mol_r = 0
      for group, occurence in self.subgroups.items():
        subgroup_r = self.model.RQ.loc[group, "R"]
        mol_r += occurence * subgroup_r
      return mol_r

    @property
    def q(self):
      r"""
      Calculates the molar van der Waals surface area parameter (:math:`q_i`) for molecule :math:`i`, from
      the summation of surface area parameters (:math:`Q_j`) for subgroup :math:`j` multiplied by their occurrences.

      .. math::
          q_i = \sum_j \nu_{ji} Q_j

      :return: molar van der Waals surface area parameter.
      :rtype: float
      """
      mol_q = 0
      for group, occurence in self.subgroups.items():
        subgroup_q = self.model.RQ.loc[group, "Q"]
        mol_q += occurence * subgroup_q
      return mol_q

    @property
    def subgroup_q(self):
      r"""
      Maps the surface area parameter (:math:`Q_j`) for each subgroup :math:`j` in molecule :math:`i` to its corresponding subgroup ID. 
      
      :return: A dictionary of subgroup IDs and their Q values.
      :rtype: dict[int, float]
      """
      q_dict = {}
      for group, occurence in self.subgroups.items():
        group_num = self.model.subgroups.loc[group, "subgroup_id"]
        q = self.model.RQ.loc[group, "Q"]
        q_dict[group_num] = q
      return q_dict

