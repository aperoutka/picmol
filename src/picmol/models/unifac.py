import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import constants
import copy

from .unifac_subgroups.unifac_subgroup_parameters import UFIP, UFSG, UFILIP, UFILSG, UFKBIIP, UFKBISG, UNIFAC_subgroup
from .unifac_subgroups.fragmentation import Groups
from .cem.point_discretization import PointDisc

def unifac_version_mapping(map_type: str):
  r"""
  Map UNIFAC version parameters to their corresponding ID.

  This function retrieves a dictionary that maps UNIFAC version parameters
  to their corresponding IDs or data structures. The specific mapping
  depends on the provided ``map_type``.

  :param map_type: identifier to retrieve the desired mapping dictionary. Valid options are:

      * `version`:  dictionary with UNIFAC version strings as keys and version IDs as values.
      * `subgroup`: dictionary with version IDs as keys and subgroup parameter objects as values.
      * `interaction`: dictionary with version IDs as keys and interaction parameter objects as values.

  :type map_type: str
  :return: dictionary containing the mapped UNIFAC version parameters.
  :rtype: dict
  """
  # get unifac version id from str
  version_str_to_id = {
    'unifac': 0,
    'unifac-il': 1,
    'unifac-kbi': 2
  }
  # get subgroup paramters for unifac version
  subgroup_dict = {
    0: UFSG,
    1: UFILSG,
    2: UFKBISG
  }
  # get interaction parameters for unifac version
  interaction_dict = {
    0: UFIP,
    1: UFILIP,
    2: UFKBIIP
  }
  # map a key to different dictionaries
  dict_map = {
    'version': version_str_to_id, 
    'subgroup': subgroup_dict,
    'interaction': interaction_dict  
  }
  return dict_map[map_type]

def get_unifac_version(version_str: str, smiles=None):
  r"""
  Determines if ionic liquids are present and sets the UNIFAC version accordingly.
  It prioritizes detecting "unifac-il" if any of the SMILES strings contain a '.' (indicating an ionic liquid).

  :param version_str: a string indicating the UNIFAC version (e.g.,
                      "unifac", "unifac-il", "unifac-kbi"). Case-insensitive.
  :type version_str: str
  :param smiles: an optional list of SMILES strings to check if ionic liquids are present.
  :type smiles: list, optional
  :returns: A string representing the determined UNIFAC version
            ("unifac", "unifac-il", or "unifac-kbi").
  :rtype: str
  """
  version_str = version_str.lower()
  # first check if IL molecule in smiles
  if smiles is not None:
    version = ['il' for smile in smiles if '.' in smile]
    if len(version) > 0:
      return 'unifac-il'
  # if not found proceed with finding version from string
  if ("md" in version_str) or ("kbi" in version_str):
    return "unifac-kbi"
  elif "il" in version_str:
    return "unifac-il"
  else:
    return "unifac"


class UNIFAC:

  r"""
  UNIFAC (UNIversal Functional Activity Coefficient) model class.

  This class implements the UNIFAC activity coefficient model for calculating
  thermodynamic properties of multi-molecule mixtures.

  :param T: temperature (K)
  :type T: float
  :param smiles: list of smiles string representing molecules in mixture.
  :type smiles: list
  :param z: composition array (mol fractions). If None, it defaults to values from PointDisc.
  :type z: numpy.ndarray, optional
  :param version: UNIFAC version to use for interaction data, as well as R and Q parameters. Options include:

    * `unifac`: standard UNIFAC
    * `unifac-il`: UNIFAC for ionic liquids
    * `unifac-kbi`: UNIFAC with regressed parameters from KBI analysis

  :type version: str

  :ivar interaction_data: dictionary containing interaction parameters for the specified UNIFAC version.
  :vartype interaction_data: dict[int, dict[int, float]]
  :ivar subgroup_data: dictionary containing subgroup parameters for the specified UNIFAC version.
  :vartype subgroup_data: dict[int, UNIFAC_subgroup]
  """

  def __init__(self, T: float, smiles: list, z = None, version = "unifac"):

    self.T = T
    self.Rc = constants.R / 1000
    self.smiles = smiles 

    num_pts = {2:10, 3:7, 4:6, 5:5, 6:4}
    self.num_comp = len(smiles)
    self.rec_steps = num_pts[self.num_comp]

    if z is not None:
      self.z = z
    else:
      point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.rec_steps, load=True, store=False)
      self.z = point_disc.points_mfr
    
    # get unifac version
    version_str = get_unifac_version(version, self.smiles)
    # get a numeric key for unifac version
    self.version = unifac_version_mapping('version')[version_str]

    # get interaction data dictionary
    self.interaction_data = unifac_version_mapping('interaction')[self.version]
    # get subgroup data dictionary
    self.subgroup_data = unifac_version_mapping('subgroup')[self.version]
  
  def unique_groups(self):
    r"""
    Identifies the unique subgroups present in the mixture.

    :return: array of unique subgroup IDs
    :rtype: numpy.ndarray
    """
    try:
      self._subgroups
    except AttributeError:
      self.subgroups()
    all_subgroups = []
    if type(self._subgroups) == list:
      for mol in self._subgroups:
        all_subgroups.extend(list(mol.keys()))
    elif type(self._subgroups) == dict:
      all_subgroups.extend(list(self._subgroups.keys()))
    self._unique_groups = np.unique(all_subgroups)
    return self._unique_groups

  @property
  def M(self):
    r"""
    :return: number of unique subgroups in the mixture
    :rtype: int
    """
    try:
      self._unique_groups
    except AttributeError:
      self.unique_groups()
    return len(self._unique_groups)
  
  @property
  def N(self):
    r"""
    :return: number of molecules in the mixture
    :rtype: int
    """
    return len(self.smiles)

  @property 
  def zc(self):
    r"""
    :return: coordination number (set to 10)
    :rtype: int
    """
    return 10

  def subgroups(self):
    r"""
    Get the subgroups present in each molecule.

    This function determines the UNIFAC subgroups from molecule
    SMILES strings, using the :class:`Groups` class to perform subgroup fragmentation.

    :returns: list of dictionaries. Each dictionary represents the subgroups
              found in the corresponding molecule, with subgroup IDs as keys
              and their occurrences as values.
    :rtype: list[dict[int, int]]
    """
    subgroup_nums = []
    for smiles in self.smiles:
      group_obj = Groups(smiles, "smiles")
      if '.' in smiles:
        subgroup_nums.append(group_obj.unifac_IL.to_num)
      else:
        subgroup_nums.append(group_obj.unifac.to_num)
    self._subgroups = subgroup_nums
    return self._subgroups

  def r(self):
    r"""
    Calculates the relative size of a molecule. 
    This property is determined by summing the product of each subgroup's r parameter (:math:`r_k`) and its frequency (:math:`\nu_{ki}`) within molecule :math:`i`.

    .. math::
        r_i = \sum_k^N \nu_{ki} r_k

    :returns: array of r parameters for each molecule
    :rtype: numpy.ndarray
    """
    try:
      self._subgroups
    except AttributeError:
      self.subgroups()
    rs = np.zeros(self.N)
    for i, mol_subgroup in enumerate(self._subgroups):
      rs[i] = sum([occurance * self.subgroup_data[group].R for group, occurance in mol_subgroup.items()])
    self._r = rs
    return self._r

  def q(self):
    r"""
    Calculates the relative surface area of a molecule. 
    This property is determined by summing the product of each subgroup's q parameter (:math:`q_k`) and its frequency (:math:`\nu_{ki}`) within molecule :math:`i`.

    .. math::
        q_i = \sum_k^N \nu_{ki} q_k
    
    :returns: array of q parameters for each molecule
    :rtype: numpy.ndarray
    """
    try:
      self._subgroups
    except AttributeError:
      self.subgroups()
    qs = np.zeros(self.N)
    for i, mol_subgroup in enumerate(self._subgroups):
      qs[i] = sum([occurance * self.subgroup_data[group].Q for group, occurance in mol_subgroup.items()])
    self._q = qs
    return self._q

  def occurance_matrix(self):
    r"""
    Generates a matrix representing subgroup occurrences in each molecule, where each row represents a unique subgroup and each column corresponds to a molecule in the mixture. 
    Each element, :math:`\nu_{ji}`, indicates the number of times subgroup :math:`j` appears in molecule :math:`i`.
   
    :return: subgroup occurrence matrix
    :rtype: numpy.ndarray
    """
    try:
      self._unique_groups
    except:
      self.unique_groups()
    occurance_matrix = np.zeros((self.M, self.N))
    for i, mol in enumerate(self._subgroups):
      subgroup_arr = np.array(list(mol.keys()))
      occurance_arr = np.array(list(mol.values()))
      for g, group in enumerate(subgroup_arr):
        occurance_matrix[self._unique_groups == group, i] = occurance_arr[g]
    self._occurance_matrix = occurance_matrix
    return self._occurance_matrix

  def subgroup_matrix(self):
    r"""
    Generates a matrix representing subgroup IDs for each molecule, where rows correspond to unique subgroups
    and columns correspond to molecules. Each element (:math:`j`, :math:`i`) represents the
    ID of subgroup :math:`j` in molecule :math:`i`. If a molecule does not contain a
    particular subgroup, the corresponding element is 0.
    
    :return: subgroup ID matrix
    :rtype: numpy.ndarray
    """
    try:
      self._unique_groups
    except:
      self.unique_groups()
    subgroup_matrix = np.zeros((self.M, self.N))
    for i, mol in enumerate(self._subgroups):
      subgroup_arr = np.array(list(mol.keys()))
      occurance_arr = np.array(list(mol.values()))
      for g, group in enumerate(subgroup_arr):
        subgroup_matrix[self._unique_groups == group, i] = subgroup_arr[g]
    self._subgroup_matrix = subgroup_matrix
    return self._subgroup_matrix
  
  def Q_matrix(self):
    r"""
    Generates a matrix of subgroup area parameters (:func:`q` values), with the same shape as :func:`subgroup_matrix`, where
    each element (:math:`j`, :math:`i`) represents the ID of subgroup :math:`j` in molecule :math:`i`.

    :return: array of subgroup Q values.
    :rtype: numpy.ndarray
    """
    try:
      self._subgroup_matrix
    except AttributeError:
      self.subgroup_matrix()
    Q_matrix = np.zeros(np.shape(self._subgroup_matrix))
    for row in range(np.shape(self._subgroup_matrix)[0]):
      for col in range(np.shape(self._subgroup_matrix)[1]):
        if self._subgroup_matrix[row,col] != 0.:
          Q_matrix[row,col] = self.subgroup_data[self._subgroup_matrix[row,col]].Q
    self._Q_matrix = Q_matrix
    return self._Q_matrix  
  
  def psis(self):
    r"""
    Calculates a *matrix* of UNIFAC interaction parameters for *all 
    possible pairs* within a given set of unique subgroups.
    This function computes the :math:`\psi` matrix, where each element, :math:`\psi_{ij}`, represents the
    interaction parameter between two subgroups. 

    .. math::
      \psi_{ij} = \exp \left( \frac{-a_{ij}}{T} \right)

    where:

      * :math:`a_{ij}` is the interaction energy parameter between subgroups :math:`i` and :math:`j`
      * :math:`\psi_{ij}` is the corresponding interaction parameter

    :return: array of interaction parameters between unique subgroups present
    :rtype: numpy.ndarray
    """
    try:
      self._unique_groups
    except AttributeError: 
      self.unique_groups()
    # map subgorup IDs to main group IDs
    subgroup_to_main = {s: self.subgroup_data[s].main_group_id for s in self._unique_groups}
    # create index mapping for main group IDs
    main_group_ids = np.array([subgroup_to_main[s] for s in self._unique_groups])
    # Create 2D grids of main group IDs for all pairs
    main1_grid, main2_grid = np.meshgrid(main_group_ids, main_group_ids, indexing='ij')
    # Initialize the psi matrix with ones
    psi_matrix = np.ones_like(main1_grid, dtype=float)
    # Identify same group interactions
    same_group = main1_grid == main2_grid
    a_matrix = np.zeros_like(main1_grid, dtype=float)
    # Create masks for unique pairs of main groups
    unique_pairs = np.unique(np.stack((main1_grid[~same_group], main2_grid[~same_group]), axis=1), axis=0)
    # Assign interaction parameters to the matrices
    for m1, m2 in unique_pairs:
        mask = (main1_grid == m1) & (main2_grid == m2)
        try:
            a_val = self.interaction_data[m1][m2]
            a_matrix[mask] = a_val
        except KeyError:
            # Missing interaction parameters; psi remains as 1.0
            pass
    # Compute psi values where main groups are different
    psi_matrix[~same_group] = np.exp(-a_matrix[~same_group] / self.T)
    # psi_matrix already has ones on the diagonal for same group interactions
    self._psis = psi_matrix
    return self._psis

  def group_counts(self):
    r"""
    Calculates the total occurrences of each unique subgroup
    across all molecules in the mixture.

    :return: array of counts for each unique subgroup
    :rtype: numpy.ndarray
    """
    try:
      self._unique_groups
    except AttributeError:
      self.unique_groups()
    group_counts = {}
    for i in range(len(self._unique_groups)):
      for group, occurrance in self._subgroups[i].items():
        group_counts[group] += occurrance
    self._group_counts = np.array(list(group_counts.values()))
    return self._group_counts
  
  def weighted_number(self):
    r"""
    Weighted mol fraction matrix, with elements :math:`W_{ijk}`, 
    where each element represents the mol fraction of molecule :math:`i`, 
    weighted by the occurrence of subgroup :math:`j` in molecule :math:`k`.

    .. math::
        W_{ijk} = x_i \nu_{jk}

    :return: matrix of subgroup occurrence weighted mol fractions
    :rtype: numpy.ndarray
    """
    try:
      self._occurance_matrix
    except AttributeError:
      self.occurance_matrix()  
    self._weighted_number = self.z[:,np.newaxis,:] * self._occurance_matrix[np.newaxis,:,:]
    return self._weighted_number


  def group_X(self):
    r"""
    Mol fraction of subgroup :math:`j` in the mixture (:math:`X_j`) considering the contributions from all molecules.

    .. math::
        X_j = \frac{\sum_i^N x_i \nu_{ji}}{\sum_i^N \sum_k^M x_i \nu_{ki}}

    :return: matrix of subgroup mol fractions
    :rtype: numpy.ndarray
    """
    try:
      self._weighted_number
    except AttributeError:
      self.weighted_number()  
    try:
      self._unique_groups
    except AttributeError:
      self.unique_groups()
    # get the sum of each array in weighted_number
    sum_weights = np.sum(self._weighted_number, axis=(2,1))
    frac_weights = self._weighted_number / sum_weights[:,np.newaxis, np.newaxis]
    # create a matrix of unique groups x number of compositions
    group_X = np.zeros((np.shape(self.z)[0], len(self._unique_groups)))
    for i in range(np.shape(frac_weights)[0]):
      for g, group in enumerate(self._unique_groups):
        group_X[i,g] = np.sum(frac_weights[i][self._unique_groups == group])
    self._group_X = group_X
    return self._group_X
  
  def X_pure(self):
    r"""
    Calculates the fraction of subgroup :math:`j` for molecule :math:`i` in the entire mixture.

    .. math::
      X_{ji} = \frac{\nu_{ji}}{\sum_i^N \sum_k^M \nu_{ki}}

    :return: matrix of subgroup fractions
    :rtype: numpy.ndarray
    """
    try:
      self._occurance_matrix
    except AttributeError:
      self.occurance_matrix()
    self._X_pure = self._occurance_matrix / np.sum(self._occurance_matrix, axis=0)
    return self._X_pure

  def group_Q(self):
    r"""
    Retrieves :math:`Q_i`, the :func:`q` parameter for each
    unique subgroup from ``subgroup_data``.

    :return: array of :math:`Q_i` values for each unique subgroup
    :rtype: numpy.ndarray
    """
    try:
      self._unique_groups
    except AttributeError:
      self.unique_groups()  
    group_Q = np.zeros(len(self._unique_groups))
    for s, subgroup in enumerate(self._unique_groups):
      group_Q[s] = self.subgroup_data[subgroup].Q
    self._group_Q = group_Q
    return self._group_Q

  def Thetas(self):
    r"""
    Calculates the area fraction, :math:`\Theta_i`, for subgroup :math:`i` in the entire mixture.

    .. math::
      \Theta_i = \frac{X_i Q_i}{\sum_j^N X_j Q_j}

    :return: matrix of area terms for each group in each composition
    :rtype: numpy.ndarray
    """
    try:
      self._group_Q
    except AttributeError:
      self.group_Q()
    try: 
      self._group_X
    except AttributeError:
      self.group_X()
    # instead of going by each molecule, this requires going by each subgroup composition
    Thetas = (self._group_X * self._group_Q).T / (self._group_X @ self._group_Q) 
    self._Thetas = Thetas.T # change to composition x groups
    return self._Thetas

  def Thetas_pure(self):
    r"""
    Calculates the subgroup area fraction, :math:`\Theta_{ji}`, for subgroup :math:`j` in molecule :math:`i`.

    .. math::
      \Theta_{ji} = \frac{X_{ji} Q_j}{\sum_k^M X_{ki} Q_l}

    :return: matrix of subgroup fractions
    :rtype: numpy.ndarray
    """
    """area group fractions for each molecule"""
    try:
      self._Q_matrix
    except AttributeError:
      self.Q_matrix()
    try:
      self._X_pure
    except AttributeError:
      self.X_pure()
    self._Thetas_pure = (self._X_pure * self._Q_matrix) / np.sum(self._X_pure * self._Q_matrix, axis=0)
    return self._Thetas_pure

  def rbar(self):
    r"""
    Calculates the linear combination of the :func:`r` parameters for each molecule, :math:`\overline{r}`.

    .. math::
        \overline{r} = \sum_j^N x_i r_i

    :return: linear combination of :func:`r`
    :rtype: numpy.ndarray
    """
    try:
      self._r
    except AttributeError:
      self.r()
    self._rbar = self.z @ self._r # takes the dot product
    return self._rbar
  
  def Vis(self):
    r"""
    Calculates the volume term, :math:`V_i`, for molecule :math:`i` in the mixture.

    .. math::
        V_i = \frac{r_i}{\sum_j^N x_j r_j}

    :return: array of volume terms for each molecule
    :rtype: numpy.ndarray
    """
    try:
      self._rbar
    except AttributeError:
      self.rbar()
    self._Vis = self._r / self._rbar[:,np.newaxis] # divides each row in r by column in rbar
    return self._Vis 

  def phis(self):
    r"""
    Calculates :math:`\phi_i`, the volume term (:func:`Vis`) weighted by mol fraction of molecule :math:`i`.

    .. math::
        \phi_i = \frac{x_i r_i}{\sum_j^N x_j r_j}

    :return: array of weighted volume terms for each molecule
    :rtype: numpy.ndarray
    """
    try:
      self._Vis
    except AttributeError:
      self.Vis()
    self._phis = self._Vis * self.z
    return self._phis
  
  def qbar(self):
    r"""
    Calculates the linear combination of the :func:`q` parameters for each molecule, :math:`\overline{q}`.

    .. math::
        \overline{q} = \sum_j^N x_i q_i

    :return: linear combination of :func:`q`
    :rtype: numpy.ndarray
    """
    try:
      self._q
    except AttributeError:
      self.q()
    self._qbar = self.z @ self._q # takes the dot product
    return self._qbar

  def Ais(self):
    r"""
    Calculates the area term, :math:`A_i`, for molecule :math:`i` in the mixture.

    .. math::
        A_i = \frac{q_i}{\sum_j^N x_j q_j}

    :return: array of area terms for each molecule
    :rtype: numpy.ndarray
    """
    try:
      self._qbar
    except AttributeError:
      self.qbar()
    self._Ais = self._q / self._qbar[:,np.newaxis] # divides each row in q by column in qbar
    return self._Ais

  def thetas(self):
    r"""
    Calculates :math:`\theta_i`, the area term (:func:`Ais`) weighted by mol fraction of molecule :math:`i`.

    .. math::
      \theta_i = \frac{x_i q_i}{\sum_j^N x_j q_j}

    :return: array of weighted area terms for each molecule
    :rtype: numpy.ndarray
    """
    try:
      self._Ais
    except AttributeError:
      self.Ais()
    self._thetas = self._Ais * self.z
    return self._thetas

  def gammas(self):
    r"""
    Total activity coefficients of molecule :math:`i` in a mixture. These coefficients are the sum of combinatorial (:math:`\gamma_i^c`) and residual (:math:`\gamma_i^r`) contributions, as shown below:

    .. math::
      \gamma_i = \gamma_i^c + \gamma_i^r

    :return: array of activity coefficients
    :rtype: numpy.ndarray
    """
    try:
      self._lngammas_c
    except AttributeError:
      self.lngammas_c()
    try:
      self._lngammas_r
    except AttributeError:
      self.lngammas_r()
    self._gammas = np.exp(self._lngammas_c + self._lngammas_r)
    return self._gammas

  def lngammas_c(self):
    r"""
    Calculates the combinatorial contribution (:math:`\gamma_i^c`) to the activity coefficient of molecule :math:`i` in a mixture.
    This accounts for differences in molecular size and shape between molecules.

    .. math::
      \ln \gamma_i^c = 1 - V_i + \ln V_i - 5 q_i \left( 1 - \frac{V_i}{A_i} + \ln \frac{V_i}{A_i} \right)

    :return: array of combinatorial activity coefficients
    :rtype: numpy.ndarray
    """
    try:
      self._q
    except AttributeError:
      self.q()
    try:
      self._r
    except AttributeError:
      self.r()
    try:
      self._thetas
    except AttributeError:
      self.thetas()
    try:
      self._phis
    except AttributeError: 
      self.phis()

    self._lngammas_c = 1 - self.Vis() + np.log(self.Vis()) - 5 * self._q * (1 - (self.Vis()/self.Ais()) + np.log(self.Vis()/self.Ais()))
    return self._lngammas_c
  
  def lngammas_r(self):
    r"""
    Calculates residual (:math:`\gamma_i^r`) contribution to activity coefficients of molecule :math:`i` in mixture. 
    This accounts for the difference compared to the residual activity coefficient of subgroup :math:`k` in the pure component reference state of molecule :math:`i`.

    .. math::
      \ln \gamma_i^r = \sum_k^M \nu_{ki} \left( \ln \Gamma_k - \ln \Gamma_{ki}  \right)

    :return: array of residual activity coefficients
    :rtype: numpy.ndarray    
    """
    try:
      self._lnGammas_subgroups
    except AttributeError:
      self.lnGammas_subgroups()
    try:
      self._lnGammas_subgroups_pure
    except AttributeError:
      self.lnGammas_subgroups_pure()
    
    self._lngammas_r = np.sum(self._occurance_matrix * (self._lnGammas_subgroups[:,np.newaxis] - self._lnGammas_subgroups_pure), axis=2)[:,0,:]
    return self._lngammas_r

  def lnGammas_subgroups_pure(self):
    r"""
    Calculates the residual activity coefficient (:math:`\Gamma_{ki}`) of subgroup :math:`k` in a reference solution consisting solely of molecules of type :math:`i`. In this method, :math:`\Theta` values come from :func:`Thetas_pure` method, where each molecule is assumed to be a pure component.

    .. math::
      \ln \Gamma_{ki} = Q_k \left( 1 - \ln \sum_i^N \Theta_i \psi_{ik} - \sum_i^N \frac{\Theta_i \psi_{ki}}{\sum_j^N \Theta_j \psi_{ji}}  \right)

    :return: array of residual contributions for the subgroup reference state
    :rtype: numpy.ndarray
    """
    try:
      self._Thetas_pure
    except AttributeError:
      self.Thetas_pure()
    try:
      self._Q_matrix
    except AttributeError:
      self.Q_matrix()
    try:
      self._psis
    except AttributeError: 
      self.psis()

    thetas_psis_12 = (self._Thetas_pure[:,np.newaxis,:] * self._psis[:,:,np.newaxis]).sum(axis=0)
    thetas_psis_21 = self._Thetas_pure[np.newaxis,:,:] * self._psis[:,:,np.newaxis]
    sum_thetas_psis_12_thetas_psis_21 = (thetas_psis_21/thetas_psis_12).sum(axis=1)
    self._lnGammas_subgroups_pure = self._Q_matrix * (1 - np.log(thetas_psis_12) - sum_thetas_psis_12_thetas_psis_21)
    return self._lnGammas_subgroups_pure

  def lnGammas_subgroups(self):
    r"""
    Calculates the residual activity coefficient (:math:`\Gamma_{k}`) of subgroup :math:`k` in a mixture.

    .. math::
      \ln \Gamma_{k} = Q_k \left( 1 - \ln \sum_i^N \Theta_i \psi_{ik} - \sum_i^N \frac{\Theta_i \psi_{ki}}{\sum_j^N \Theta_j \psi_{ji}}  \right)

    :return: array of residual contributions for each subgroup
    :rtype: numpy.ndarray
    """
    try:
      self._Thetas
    except AttributeError:
      self.Thetas()
    try:
      self._psis
    except AttributeError: 
      self.psis()
    try:
      self._subgroup_matrix
    except:
      self.subgroup_matrix()

    Theta_psi = self._Thetas @ self._psis # shape: (S, G)

    # expand dimensions for broadcasting
    Theta_m = self._Thetas[:,np.newaxis,:] # shape: (S, 1, G)
    psi_km = self._psis[np.newaxis,:,:] # shape: (1, G, G)
    Theta_psi_nm = Theta_psi[:,np.newaxis,:] # shape: (S, 1, G)

    # compute theta * psi / (sum{theta*psi})
    Theta_psi_km = Theta_m * psi_km # shape: (S, G, G)
    sum_Theta_psi_km_Theta_psi_nm = (Theta_psi_km / Theta_psi_nm).sum(axis=2) # shape: (S, G)

    # compute the residual activity coef. for each group in each molecule
    subgroup_lnGammas_subgroups = self._group_Q[np.newaxis,:] * (1 - np.log(Theta_psi) - sum_Theta_psi_km_Theta_psi_nm) # shape: (S, G)

    # convert to appropriate dimensions
    subgroup_lnGammas_subgroups_matrix = np.zeros(np.shape(self._weighted_number))
    for i in range(np.shape(self._weighted_number)[0]):
      for g, group in enumerate(self._unique_groups):
        subgroup_lnGammas_subgroups_matrix[i][self._subgroup_matrix == group] = subgroup_lnGammas_subgroups[i,g]

    self._lnGammas_subgroups = subgroup_lnGammas_subgroups_matrix
    return self._lnGammas_subgroups

  def GE(self):
    r"""
    Gibbs excess energy of the mixture.

    .. math::
      \frac{G^E}{RT} = \sum_{i}^N x_i \ln \gamma_i

    :return: array of Gibbs excess energy
    :rtype: numpy.ndarray
    """
    try:
      self._gammas
    except AttributeError: 
      self.gammas()
    self._GE = (self.Rc * self.T * np.sum(self.z * np.log(self._gammas), axis=1))
    return self._GE

  def Hmix(self):
    try:
      self._gammas
    except AttributeError: 
      self.gammas()
    return self.Rc * self.T * np.sum(self.z * self.lngammas_r(), axis=1)

  def Smix(self):
    try:
      self._gammas
    except AttributeError: 
      self.gammas()
    return self.Rc * self.T * np.sum(self.z * (self.lngammas_c() + np.log(self.z)) , axis=1)

  def GM(self):
    r"""
    Gibbs mixing free energy of the mixture.

    .. math::
      \frac{\Delta G_{mix}}{RT} = \sum_{i}^N x_i \ln \left( x_i \gamma_i \right)

    :return: array of Gibbs mixing free energy
    :rtype: numpy.ndarray
    """
    try:
      self._gammas
    except AttributeError: 
      self.gammas()
    self._GM = (self.Rc * self.T * np.sum(self.z * np.log(self._gammas * self.z), axis=1))
    return self._GM
  
  def dlngammas_c_dxs(self):
    r"""
    Calculates the derivative of the combinatorial contribution to the activity coefficients with respect to the mol fractions.

    .. math::
      \frac{\partial \ln \gamma_i^c}{\partial x_j} = -5 q_i \left( \frac{\frac{\partial V_i}{\partial x_j}}{V_i} - \frac{V_i \frac{\partial A_i}{\partial x_j}}{A_i^2} \frac{A_i}{V_i} - \frac{\frac{\partial V_i}{\partial x_j}}{A_i} + \frac{V_i \frac{\partial A_i}{\partial x_j}}{A_i^2} \right) - \frac{\partial V_i}{\partial x_j} + \frac{\frac{\partial V_i}{\partial x_j}}{V_i}

    :return: Array of the derivatives of the combinatorial activity coefficients with respect to mol fractions
    :rtype: numpy.ndarray    
    """

    try:
      self._dVis_dxs
    except AttributeError: 
      self.dVis_dxs()
    try:
      self._dAis_dxs
    except AttributeError: 
      self.dAis_dxs()
    try:
      self._Vis
    except AttributeError: 
      self.Vis()
    try:
      self._Ais
    except AttributeError: 
      self.Ais()
  
    Vis = self._Vis[:,np.newaxis,:]
    Ais = self._Ais[:,np.newaxis,:]
    dlngammas_c_dxs = -5*self._q* ((self._dVis_dxs/Vis)-(Vis*self._dAis_dxs/Ais**2)*(Ais/Vis) - (self._dVis_dxs/Ais) + (Vis*self._dAis_dxs/Ais**2)) - self._dVis_dxs + (self._dVis_dxs/Vis)
    self._dlngammas_c_dxs = dlngammas_c_dxs
    return self._dlngammas_c_dxs

  def dlngammas_r_dxs(self):
    r"""
    Calculates the derivative of the residual contribution to the activity coefficients with respect to the mol fractions.

    .. math::
      \frac{\partial \ln \gamma_i^r}{\partial x_j} = \sum_k^M \nu_{ki} \frac{\partial \ln \Gamma_k}{\partial x_j}

    :return: array of the derivatives of the residual activity coefficients with respect to mol fractions
    :rtype: numpy.ndarray
    """
    try:
      self._occurance_matrix
    except AttributeError: 
      self.occurance_matrix()
    try:
      self._dlnGammas_subgroups_dxs 
    except AttributeError: 
      self.dlnGammas_subgroups_dxs()

    dlngammas_r_dxs = self._occurance_matrix.T @ self._dlnGammas_subgroups_dxs
    dlngammas_r_dxs = dlngammas_r_dxs.reshape(np.shape(self.z)[0], self.N, self.N)
    self._dlngammas_r_dxs = dlngammas_r_dxs
    return self._dlngammas_r_dxs

  def dAis_dxs(self):
    r"""
    Calculates the derivative of the surface area fractions with respect to the mol fractions.

    .. math::
      \frac{\partial A_i}{\partial x_j} = -\frac{q_i q_j}{\overline{q}^2}
    
    :return: array of the derivatives of the surface area term with respect to mol fractions
    :rtype: numpy.ndarray
    """
    try:
      self._qbar
    except AttributeError: 
      self.Ais()
    
    self._dAis_dxs = (-np.outer(self._q, self._q)) / self._qbar[:,np.newaxis,np.newaxis]**2
    return self._dAis_dxs

  def dVis_dxs(self):
    r"""
    Calculates the derivative of the volume fractions with respect to the mol fractions.

    .. math::
      \frac{\partial V_i}{\partial x_j} = -\frac{r_i r_j}{\overline{r}^2}

    :return: array of the derivatives of the volume term with respect to mol fractions
    :rtype: numpy.ndarray
    """
    try:
      self._rbar
    except AttributeError: 
      self.rbar()
    
    self._dVis_dxs = (-np.outer(self._r, self._r)) / self._rbar[:,np.newaxis,np.newaxis]**2
    return self._dVis_dxs
  

  def F(self):
    r"""
    Calculates the inverse of a weighted sum of molecule :math:`i`, where the sum is over the occurance of subgroups :math:`j` in molecules :math:`k` within the mixture.

    .. math::
      F = \frac{1}{\sum_i^N \sum_j^M x_i \nu_{ji}}

    :return: :math:`F` parameter of the mixture
    :rtype: float
    """
    try:
      self._weighted_number
    except AttributeError: 
      self.weighted_number()
    self._F = 1/(self._weighted_number.sum(axis=1)).sum(axis=1) 
    return self._F

  def G(self):
    r"""
    Calculates a parameter, :math:`G`, related to the group contributions in the mixture.
    Specifically, :math:`G`, is the inverse of the sum of the products of :math:`X_k` and :math:`Q_k` for all subgroups in the mixture.

    .. math::
      G = \frac{1}{\sum_j^M X_j Q_j}

    :return: The calculated parameter :math:`G`.
    :rtype: numpy.ndarray
    """
    try:
      self._group_X
    except AttributeError:
      self.group_X()
    try:
      self._group_Q
    except AttributeError: 
      self.group_Q()
    self._G = 1/(self._group_X * self._group_Q).sum(axis=1)
    return self._G
  
  def sum_occurance_matrix(self):
    r"""
    Calculates the total number of subgroups in molecule :math:`i`.

    .. math::
      \left( \nu \right)_{sum,i} = \sum_j^M \nu_{ji}

    :return: array of number of subgroups in a molecule
    :rtype: numpy.ndarray
    """
    try:
      self._occurance_matrix
    except AttributeError: 
      self.occurance_matrix()
    self._sum_occurance_matrix = self._occurance_matrix.sum(axis=0)
    return self._sum_occurance_matrix

  def sum_weighted_number(self):
    r"""
    Calculates the total weighted number of molecule :math:`i` with occurance of subgroup :math:`j` for all molecules in mixture.

    .. math::
      (\nu x)_{sum,k} = \sum_k^N W_{ijk}

    :return: array of weighted sums of molecule :math:`i`
    :rtype: numpy.ndarray
    """
    try:
      self._weighted_number
    except AttributeError: 
      self.weighted_number()
    self._sum_weighted_number = self._weighted_number.sum(axis=2)
    return self._sum_weighted_number

  def dThetas_dxs(self):
    r"""
    Calculates the derivative of the surface area term (:math:`\Theta_i`) with respect to the mol fractions (:math:`x_j`).

    .. math::
      \frac{\partial \Theta_i}{\partial x_j} = F G Q_i \left( F G (\nu x)_{sum,i} \left( \sum_k^M F Q_k \left( \nu \right)_{sum,j} - \sum_k^M Q_k \nu_{jk} \right) - F \left( \nu \right)_{sum,j} (\nu x)_{sum,i} + \nu_{ji} \right)

    :return: array of the derivatives of the surface area fractions with respect to mole fractions.
    :rtype: numpy.ndarray
    """
    try:
      self._F
    except AttributeError:
      self.F()
    try:
      self._G
    except AttributeError:
      self.G()
    try:
      self._sum_occurance_matrix
    except AttributeError: 
      self.sum_occurance_matrix()
    try:
      self._sum_weighted_number
    except AttributeError: 
      self.sum_weighted_number()

    lenF = self._F.shape[0] # Number of F and G values

    # Initialize output arrays 
    dThetas_dxs = np.zeros((lenF, self.M, self.N))
    vec0 = np.zeros((lenF, self.N))

    # Compute tot0: F multiplied by the sum over N_groups of sum_weighted_number * self._group_Q
    tot0 = self._F * np.sum(self._sum_weighted_number * self._group_Q[np.newaxis, :], axis=1)

    # Compute tot1: Negative sum over N_groups of self._group_Q * vs
    tot1 = -np.sum(self._group_Q[:, np.newaxis] * self._occurance_matrix, axis=0)

    # Compute vec0: Vector of size (lenF, N)
    # Expand dimensions for broadcasting
    tot0_expanded = tot0[:, np.newaxis]
    F_expanded = self._F[:, np.newaxis]
    G_expanded = self._G[:, np.newaxis]
    sum_occurance_matrix_expanded = self._sum_occurance_matrix[np.newaxis, :]
    tot1_expanded = tot1[np.newaxis, :]

    # Compute intermediate values
    temp = tot0_expanded * sum_occurance_matrix_expanded + tot1_expanded
    vec0 = F_expanded * (G_expanded * temp - sum_occurance_matrix_expanded)

    # Compute ci: (F * G) outer product with self._group_Q
    ci = (self._F * self._G)[:, np.newaxis] * self._group_Q[np.newaxis, :]

    # Compute the outer product of sum_weighted_number and vec0
    outer = self._sum_weighted_number[:, :, np.newaxis] * vec0[:, np.newaxis, :]

    # Broadcast vs to match dimensions
    vs_expanded = self._occurance_matrix[np.newaxis, :, :]

    # Compute dThetas_dxs
    dThetas_dxs = ci[:, :, np.newaxis] * (outer + vs_expanded)

    self._dThetas_dxs = dThetas_dxs
    return self._dThetas_dxs


  def Ws(self):
    r"""
    Calculates the weights of :math:`\psi` interaction parameters by derivative of area terms for molecule :math:`i` over occurance of subgroups :math:`j` in molecule :math:`k`.

    .. math::
      W_{ki} = \sum_j^M \psi_{jk} \frac{\partial \Theta_j}{\partial x_i}

    :return: array of weighted :math:`\psi` values by first derivative of area term
    :rtype: numpy.ndarray
    """
    try:
      self._psis
    except AttributeError:
      self.psis()
    try:
      self._dThetas_dxs
    except AttributeError:
      self.dThetas_dxs()
    self._Ws =  (self._psis[np.newaxis,:,:,np.newaxis] * self._dThetas_dxs[:,:,np.newaxis,:]).sum(axis=1)
    return self._Ws
  

  def Theta_Psi_sum_invs(self):
    r"""
    Calculates sum area terms for subgroup :math:`m` and the corresponding interaction parameter for molecule :math:`k`.
    
    .. math::
        Z_k = \frac{1}{\sum_m^M \Theta_m \psi_{mk}}

    :return: inverse sum of area term and interaction paramter for molecule :math:`k`
    :rtype: numpy.ndarray
    """
    try:
      self._Thetas
    except AttributeError: 
      self.Thetas()
    try:
      self._psis
    except AttributeError:
      self.psis()
    self._Theta_Psi_sum_invs = 1/(self._Thetas @ self._psis)
    return self._Theta_Psi_sum_invs
  

  def dlnGammas_subgroups_dxs(self):
    r"""
    Calculates the first derivative of residual activity coefficient (:math:`\Gamma_{k}`) of subgroup :math:`k` in a mixture with respect to mol fraction of molecule :math:`j`.

    .. math::
      \frac{\partial \ln \Gamma_k}{\partial x_j} = Q_k 
        \left( 
          - \frac{\sum_l^M \psi_{lk} \frac{\partial \Theta_l}{\partial x_j}}{\sum_l^M \Theta_l \psi_{lk}}
          - \sum_l^M \frac{\psi_{kl} \frac{\partial \Theta_l}{\partial x_j}}{\sum_m^M \Theta_m \psi_{ml}}
          + \sum_l^M \frac{\left( \sum_m^M \psi_{ml} \frac{\partial \Theta_m}{\partial x_j} \right) \Theta_l \psi_{kl}}{\left( \sum_m^M \Theta_m \psi_{ml} \right)^2} 
        \right)

    :return: mol fraction derivatives of :math:`\Gamma_{k}` for each subgroup
    :rtype: numpy.ndarray
    """
    try:
      self._Ws
    except AttributeError:
      self.Ws()
    try:
      self._Theta_Psi_sum_invs
    except AttributeError:
      self.Theta_Psi_sum_invs()
    
    ### Step 1: Compute the First Term
    # First_term[l, k, i] = -Ws[l, k, i] * Theta_Psi_sum_invs[l, k]
    First_term = -self._Ws * self._Theta_Psi_sum_invs[:, :, np.newaxis]  # Shape: (L, N_groups, N)

    ### Step 2: Compute Intermediate Values
    # Compute Ws_TPT_inv_Thetas[l, m, i] = Ws[l, m, i] * Theta_Psi_sum_invs[l, m] * Thetas[l, m]
    Ws_TPT_inv_Thetas =self._Ws * self._Theta_Psi_sum_invs[:, :, np.newaxis] * self._Thetas[:, :, np.newaxis]  # Shape: (L, N_groups, N)

    # Compute Delta_dThetas_dxs[l, m, i] = dThetas_dxs[m, i] - Ws_TPT_inv_Thetas[l, m, i]
    Delta_dThetas_dxs = self._dThetas_dxs[np.newaxis, :, :] - Ws_TPT_inv_Thetas  # Shape: (L, N_groups, N)

    ### Step 3: Compute the Second Term using Batch Matrix Multiplication
    # Compute A[l, k, m] = psis[k, m] * Theta_Psi_sum_invs[l, m]
    psis_sum_Theta_psis_inv = self._psis[np.newaxis, :, :] * self._Theta_Psi_sum_invs[:, np.newaxis, :]  # Shape: (L, N_groups, N_groups)

    # Compute Second_term[l, k, i] = sum over m [A[l, k, m] * Delta_dThetas_dxs[l, m, i]]
    Second_term = np.matmul(psis_sum_Theta_psis_inv, Delta_dThetas_dxs)  # Shape: (L, N_groups, N)

    ### Step 4: Compute Total and Final Result
    Total = First_term - Second_term  # Shape: (L, N_groups, N)

    # Multiply by self._group_Q to get the final result
    self._dlnGammas_subgroups_dxs = Total * self._group_Q[np.newaxis, :, np.newaxis]  # Shape: (L, N_groups, N)
    return self._dlnGammas_subgroups_dxs


  def dGE_dxs(self):
    r"""
    Calculates the first derivative of Gibbs excess energy with respect to mol fraction of molecule :math:`i`.

    .. math::
      \frac{1}{RT}\frac{\partial G^E}{\partial x_i} = \ln \gamma_i^c + \ln \gamma_i^r + \sum_j^N x_j \left( \frac{\partial \ln \gamma_j^c}{\partial x_i} + \frac{\partial \ln \gamma_j^r}{\partial x_i} \right)

    :return: first component derivative of excess Gibbs energy
    :rtype: numpy.ndarray
    """
    try:
      self._dlngammas_c_dxs
    except AttributeError: 
      self.dlngammas_c_dxs()
    try:
      self._lngammas_r_dxs
    except AttributeError: 
      self.dlngammas_r_dxs()
    try:
      self._lngammas_c
    except AttributeError:
      self.lngammas_c()
    try:
      self._lngammas_r
    except AttributeError:
      self.lngammas_r()

    lngammas = self._lngammas_c + self._lngammas_r
    dlngammas = np.sum(self.z[:,np.newaxis,:] * (self._dlngammas_c_dxs + self._dlngammas_r_dxs), axis=2)

    self._dGE_dxs = self.Rc * self.T * (lngammas + dlngammas)

    return self._dGE_dxs


  def mu(self):
    r"""
    Calculates the chemical potential of molecule :math:`i`.

    .. math::
      \frac{\mu_i}{RT} = 
        1 + \ln x_i 
        + \ln \gamma_i^c + \ln \gamma_i^r + 
        \sum_j^N x_j \left( \frac{\partial \ln \gamma_j^c}{\partial x_i} + \frac{\partial \ln \gamma_j^r}{\partial x_i} \right)

    :return: chemical potential of molecule :math:`i`
    :rtype: numpy.ndarray
    """
    try:
      self._dlngammas_c_dxs
    except AttributeError: 
      self.dlngammas_c_dxs()
    try:
      self._lngammas_r_dxs
    except AttributeError: 
      self.dlngammas_r_dxs()
    try:
      self._lngammas_c
    except AttributeError:
      self.lngammas_c()
    try:
      self._lngammas_r
    except AttributeError:
      self.lngammas_r()

    lngammas = self._lngammas_c + self._lngammas_r
    dlngammas = np.sum(self.z[:,np.newaxis,:] * (self._dlngammas_c_dxs + self._dlngammas_r_dxs), axis=2)

    self._mu = self.Rc * self.T * (lngammas + dlngammas + np.log(self.z) + 1)

    return self._mu
  

  def dGM_dxs(self):
    r"""
    Calculates first derivative of Gibbs mixing free energy with respect to mol fraction of molecule :math:`i`.

    .. math::
     \frac{\partial \Delta G_{mix}}{\partial x_i} = \mu_i - \mu_N

    :return: first component derivative of Gibbs mixing free energy
    :rtype: numpy.ndarray
    """
    
    try:
      self._mu 
    except:
      self.mu()
    
    dGM_dxs = np.zeros((self.z.shape[0], self.z.shape[1]-1))
    for i in range(self.z.shape[1]-1):
      dGM_dxs[:,i] = self._mu[:,i] - self._mu[:,self.z.shape[1]-1]
    self._dGM_dxs = dGM_dxs
    return self._dGM_dxs
  
  
  def d2lngammas_c_dxixjs(self):
    r"""
    Calculates the second mol fraction derivative with respect to molecule :math:`j` and molecule :math:`k` of combinatorial contribution to activity coefficients of molecule :math:`i`.

    .. math::
      \frac{\partial \ln \gamma^c_i}{\partial x_j \partial x_k} =
      5 q_{i} \left(\frac{- \frac{\partial^{2}V_{i}}{\partial x_{k}\partial x_{j}} + \frac{V_{i}
      \frac{\partial^{2}A_{i}}{\partial x_{k}\partial x_{j}}}{A_{i}} + \frac{\frac{\partial A_{i}}{\partial x_{j}} 
      \frac{\partial V_{i}}{\partial x_{k}}}{A_{i}} + \frac{\frac{\partial A_{i}}{\partial x_{k}} 
      \frac{\partial V_{i}}{\partial x_{j}}}{A_{i}} - \frac{2 V_{i} \frac{\partial A_{i}}{\partial x_{j}}
       \frac{\partial A_{i}}{\partial x_{k}}}{A_{i}^{2}}}{V_{i}} + \frac{\left(
      \frac{\partial V_{i}}{\partial x_{j}} - \frac{V_{i} \frac{\partial A_{i}}{\partial x_{j}}}
      {A_{i}}\right) \frac{\partial V_{i}}{\partial x_{k}}}{V_{i}^{2}}
      + \frac{\frac{\partial^{2} V_{i}}{\partial x_{k}\partial x_{j}}}{A_{i}} - \frac{\left(
      \frac{\partial V_{i}}{\partial x_{j}} - \frac{V_{i} \frac{\partial A_{i}}{\partial x_{j}}}{
      A_{i}}\right) \frac{\partial A_{i}}{\partial x_{k}}}{A_{i} V_{i}} - \frac{V_{i}
      \frac{\partial^{2} A_{i}}{\partial x_{k}\partial x_{j}}}{A_{i}^{2}} - \frac{\frac{\partial A_{i}}
      {\partial x_{j}} \frac{\partial V_{i}}{\partial x_{k}}}{A_{i}^{2}}
      - \frac{\frac{\partial A_{i}}{\partial x_{k}} \frac{\partial V_{i}}{\partial x_{j}}}{A_{i}^{2}}
      + \frac{2 V_{i} \frac{\partial A_{i}}{\partial x_{j}} \frac{\partial A_{i}}{\partial x_{k}}}
      {A_{i}^{3}}\right) - \frac{\partial^{2} V_i}{\partial x_{k}\partial x_{j}}
      + \frac{\frac{\partial^{2} V_i}{\partial x_{k}\partial x_{j}}}{V_i} - \frac{\frac{\partial  V_i}
      {\partial x_{j}} \frac{\partial  V_i}{\partial x_{k}}}{V_i^{2}}

    :return: array of second derivative of combinatorial activity coefficient with respect to mol fractions
    :rtype: numpy.ndarray
    """
    try:
      self._Vis
    except AttributeError:
      self.Vis()
    try:
      self._Ais
    except AttributeError:
      self.Ais()
    try:
      self._dVis_dxs
    except AttributeError: 
      self.dVis_dxs()
    try:
      self._dAis_dxs
    except AttributeError: 
      self.dAis_dxs()
    try:
      self._d2Vis_dxixjs
    except AttributeError: 
      self.d2Vis_dxixjs()
    try:
      self._d2Ais_dxixjs
    except AttributeError: 
      self.d2Ais_dxixjs()

    # Precompute repeated terms
    Vi_inv2 = 1.0 / (self._Vis ** 2)  # Shape: (3, N)
    x1 = 1.0 / self._Ais  # Shape: (3, N)
    x4 = x1 ** 2  # Shape: (3, N)
    Ai_inv3 = x1 ** 3  # Shape: (3, N)
    x5 = self._Vis * x4  # Shape: (3, N)
    x15 = 1.0 / self._Vis  # Shape: (3, N)
    Vi_inv2 = x15 ** 2  # Shape: (3, N)

    # Expand dimensions for broadcasting
    Vi_expanded = self._Vis[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
    qi_expanded = self._q[np.newaxis, :, np.newaxis, np.newaxis]  # Shape: (1, N, 1, 1)
    Vi_inv2_expanded = Vi_inv2[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
    x1_expanded = x1[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
    x4_expanded = x4[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
    Ai_inv3_expanded = Ai_inv3[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
    x5_expanded = x5[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
    x15_expanded = x15[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
    Vi_inv2_expanded = Vi_inv2[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)

    # Get dVis and dAis variables
    x6 = self._dAis_dxs[:, :, :, np.newaxis]  # Shape: (3, N, N, 1)
    x10 = self._dVis_dxs[:, :, :, np.newaxis]  # Shape: (3, N, N, 1)
    dVi_dxj = x10  # Shape: (3, N, N, 1)

    x7 = self._dVis_dxs[:, :, np.newaxis, :]  # Shape: (3, N, 1, N)
    dVi_dxk = x7  # Shape: (3, N, 1, N)
    x9 = self._dAis_dxs[:, :, np.newaxis, :]  # Shape: (3, N, 1, N)

    # Second derivatives
    x0 = self._d2Vis_dxixjs  # Shape: (3, N, N, N)
    x2 = x0  # Same as x0
    x3 = self._d2Ais_dxixjs  # Shape: (3, N, N, N)

    # Compute intermediate variables
    x8 = x6 * x7  # Shape: (3, N, N, N)
    x11 = x10 * x9  # Shape: (3, N, N, N)
    x12 = 2.0 * x6 * x9  # Shape: (3, N, N, N)

    x13 = Vi_expanded * x1_expanded  # Shape: (3, N, 1, 1)
    x13_x6 = x13 * x6  # Shape: (3, N, N, 1)
    x14 = x10 - x13_x6  # Shape: (3, N, N, 1)

    # Compute the value
    self._d2lngammas_c_dxixjs = (
        5.0 * qi_expanded * (
            -x1_expanded * x14 * x15_expanded * x9 + x1_expanded * x2 - x11 * x4_expanded
            + x15_expanded * (
                x1_expanded * x11 + x1_expanded * x8 - x12 * x5_expanded + x13 * x3 - x2
            )
            - x3 * x5_expanded - x4_expanded * x8 + x14 * x7 * Vi_inv2_expanded + Vi_expanded * x12 * Ai_inv3_expanded
        )
        - x0 + x0 / self._Vis[:, :, np.newaxis, np.newaxis] - dVi_dxj * dVi_dxk * Vi_inv2_expanded
    )

    # Assign the computed values to the output array
    return self._d2lngammas_c_dxixjs  # Shape: (3, N, N, N)

  def d2lngammas_r_dxixjs(self):
    r"""
    Calculates the second mol fraction derivative with respect to molecule :math:`j` and molecule :math:`k` of residual contribution to activity coefficients of molecule :math:`i`.

    .. math::
      \frac{\partial^2 \ln \gamma_i^r}{\partial x_j \partial x_k} = \sum_{m}^{M}
      \nu_m^{mi} \frac{\partial^2 \ln \Gamma_m}{\partial x_j^2}

    :return: array of second derivative of combinatorial activity coefficient with respect to mol fractions
    :rtype: numpy.ndarray
    """
    try:
      self._d2lnGammas_subgroups_dxixjs
    except AttributeError:
      self.d2lnGammas_subgroups_dxixjs()
    try:
      self._occurance_matrix
    except AttributeError:
      self.occurance_matrix()
    self._d2lngammas_r_dxixjs = np.einsum('mi,fjkm->fijk', self._occurance_matrix, self._d2lnGammas_subgroups_dxixjs)
    return self._d2lngammas_r_dxixjs


  def d2Vis_dxixjs(self):
    r"""
    Calculates the second mol fraction derivatives of volume term with respect to molecule :math:`i` and molecule :math:`j`.

    .. math::
      \frac{\partial^2 V_i}{\partial x_j \partial x_k} = \frac{2 r_i r_j r_k}{\left( \sum_l^N r_l x_l \right)^3}

    :return: second derivative of volume terms
    :rtype: numpy.ndarray
    """
    try:
      self._rbar
    except AttributeError: 
      self.rbar()
    
    rbar_sum_inv3_2 = 2.0*1/(self._rbar)**3
    rs_i = self._r[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    rs_j = self._r[np.newaxis, :, np.newaxis]  # (1, N, 1)
    rs_k = self._r[np.newaxis, np.newaxis, :]  # (1, 1, N)
    rs_3 = rs_i * rs_j * rs_k 
    self._d2Vis_dxixjs = rbar_sum_inv3_2[:,np.newaxis, np.newaxis,np.newaxis] * rs_3[np.newaxis,:]
    return self._d2Vis_dxixjs


  def d2Ais_dxixjs(self):
    r"""
    Calculates the second mol fraction derivatives of surface area term with respect to molecule :math:`i` and molecule :math:`j`.

    .. math::
      \frac{\partial^2 A_i}{\partial x_j \partial x_k} = \frac{2 q_i q_j q_k}{ \left( \sum_l^N q_l x_l \right)^3}

    :return: second derivative of surface area terms
    :rtype: numpy.ndarray
    """
    try:
      self._qbar
    except AttributeError: 
      self.qbar()
    
    qbar_sum_inv3_2 = 2.0*1/(self._qbar)**3
    qs_i = self._q[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    qs_j = self._q[np.newaxis, :, np.newaxis]  # (1, N, 1)
    qs_k = self._q[np.newaxis, np.newaxis, :]  # (1, 1, N)
    qs_3 = qs_i * qs_j * qs_k 
    self._d2Ais_dxixjs = qbar_sum_inv3_2[:,np.newaxis, np.newaxis,np.newaxis] * qs_3[np.newaxis,:]
    return self._d2Ais_dxixjs
  

  def Zs(self):
    try:
      self._Thetas
    except AttributeError:
      self.Thetas()
    try:
      self._psis
    except AttributeError:
      self.psis()
    self._Zs = 1/(self._Thetas @ self._psis)
    return self._Zs

  def d2lnGammas_subgroups_dxixjs(self):
    r"""
    Calculate the second mol fraction derivatives of the :math:`\ln \Gamma_k` (for subgroup :math:`k`) parameters for the phase with respect to molecule :math:`i` and molecule :math:`j`.

    .. math::
        \frac{\partial^2 \ln \Gamma_k}{\partial x_i \partial x_j} = -Q_k\left(
        -Z_k K_{kij} - \sum_m^M Z_m^2 K_{mij}\Theta_m \psi_{km}
        -W_{ki} W_{kj}) Z_k^2
        + \sum_m^M Z_m \psi_{km} \frac{\partial^2 \Theta_m}{\partial x_i \partial x_j}
        - \sum_m \left(W_{mj} Z_m^2 \psi_{km} \frac{\partial \Theta_m}{\partial x_i}
        + W_{mi} Z_m^2 \psi_{km} \frac{\partial \Theta_m}{\partial x_j}\right)
        + \sum_m^M 2 W_{mi} W_{mj} Z_m^3 \Theta_m \psi_{km}\right)  
    
    where: 

    .. math::
        K_{kij} = \sum_m^M \psi_{mk} \frac{\partial^2 \Theta_m}{\partial x_i \partial x_j}
    
    :return: array of second mol fraction derivatives of Gamma parameters for each subgroup
    :rtype: numpy.ndarray
    """
    try:
      self._Zs
    except AttributeError: 
      self.Zs()
    try:
      self._Ws
    except AttributeError:
      self.Ws()
    try:
      self._d2Thetas_dxixjs
    except AttributeError:
      self.d2Thetas_dxixjs()

    d2lnGammas_subgroups_dxixjs = np.empty((np.shape(self.z)[0], self.N, self.N, self.M))

    for f in range(np.shape(self.z)[0]):
      for i in range(self.N):
        # Extract the relevant slice for d2Thetas_dxixjs
        d2Thetas_dxixjs_ij = self._d2Thetas_dxixjs[f, i]  # Shape: (N, N_groups)
        
        # Compute vec0 for all j and k simultaneously
        # vec0[j, k] = sum over m of psis[m, k] * d2Thetas_dxixjs_ij[j, m]
        vec0 = np.dot(d2Thetas_dxixjs_ij, self._psis)  # Shape: (N, N_groups)
        
        for j in range(self.N):
          # Extract vectors for the current indices
          vec0_j = vec0[j]  # Shape: (N_groups,)
          d2Thetas_dxixjs_ij_j = d2Thetas_dxixjs_ij[j]  # Shape: (N_groups,)
          Ws_f_i = self._Ws[f, :, i]  # Shape: (N_groups,)
          Ws_f_j = self._Ws[f, :, j]  # Shape: (N_groups,)
          Zs_f = self._Zs[f]  # Shape: (N_groups,)
          Thetas_f = self._Thetas[f]  # Shape: (N_groups,)
          dThetas_dxs_f_i = self._dThetas_dxs[f, :, i]  # Shape: (N_groups,)
          dThetas_dxs_f_j = self._dThetas_dxs[f, :, j]  # Shape: (N_groups,)

          # Compute intermediate variables A and B
          A = 2.0 * Ws_f_i * Ws_f_j * Zs_f - vec0_j  # Shape: (N_groups,)
          B = Ws_f_j * dThetas_dxs_f_i + Ws_f_i * dThetas_dxs_f_j  # Shape: (N_groups,)

          # Compute d for all m simultaneously
          d = d2Thetas_dxixjs_ij_j + Zs_f * (A * Thetas_f - B)  # Shape: (N_groups,)

          # Compute v for all k simultaneously
          v = np.dot(self._psis, d * Zs_f)  # Shape: (N_groups,)

          # Add the remaining term to v
          v += Zs_f * (vec0_j - Ws_f_i * Ws_f_j * Zs_f)

          # Compute the final result
          d2lnGammas_subgroups_dxixjs[f,i,j, :] = -v * self._group_Q

    self._d2lnGammas_subgroups_dxixjs = d2lnGammas_subgroups_dxixjs
    return self._d2lnGammas_subgroups_dxixjs


  def d2Thetas_dxixjs(self):
    r"""
    Calculates the second derivative of the area term of subgroup :math:`i` with respect to mol fractions of molecule :math:`j` and molecule :math:`k`.

    .. math::
      \frac{\partial^2 \Theta_i}{\partial x_j \partial x_k} =
      \frac{Q_i}{\sum_n^N Q_n (\nu x)_{sum,n}}\left(
      -F(\nu)_{sum,j} \nu_{ik} - F (\nu)_{sum,k}\nu_{ij}
      + 2F^2(\nu)_{sum,j} (\nu)_{sum,k} (\nu x)_{sum,i}
      + \frac{F (\nu x)_{sum,i}\left(
      \sum_n^M(-2 F Q_n (\nu)_{sum,j} (\nu)_{sum,k}
      (\nu x)_{sum,n} + Q_n (\nu)_{sum,j} \nu_{nk} + Q_n (\nu)_{sum,k}\nu_{nj}
      )\right) }
      {\sum_n^M Q_n (\nu x)_{sum,n} }
      + \frac{2(\nu x)_{sum,i}(\sum_n^M[-FQ_n (\nu)_{sum,j} (\nu x)_{sum,n} + Q_n \nu_{nj}])
      (\sum_n^M[-FQ_n (\nu)_{sum,k} (\nu x)_{sum,n} + Q_n \nu_{nk}])  }
      {\left( \sum_n^M Q_n (\nu x)_{sum,n} \right)^2}
      - \frac{\nu_{ij}(\sum_n^M -FQ_n (\nu)_{sum,k} (\nu x)_{sum,n} + Q_n \nu_{nk} )}
      {\left( \sum_n^M Q_n (\nu x)_{sum,n} \right)}
      - \frac{\nu_{ik}(\sum_n^M -FQ_n (\nu)_{sum,j} (\nu x)_{sum,n} + Q_n \nu_{nj} )}
      {\left( \sum_n^M Q_n (\nu x)_{sum,n} \right)}
      + \frac{F(\nu)_{sum,j} (\nu x)_{sum,i} (\sum_n^M -FQ_n (\nu)_{sum,k}
      (\nu x)_{sum,n} + Q_n \nu_{nk})}
      {\left(\sum_n^M Q_n (\nu x)_{sum,n} \right)}
      + \frac{F(\nu)_{sum,k} (\nu x)_{sum,i} (\sum_n^M -FQ_n (\nu)_{sum,j}
      (\nu x)_{sum,n} + Q_n \nu_{nj})}
      {\left(\sum_n^M Q_n (\nu x)_{sum,n} \right)}
      \right)

    :return: array of second mol fraction derivative of area fractions
    :rtype: numpy.ndarray
    """

    try:
      self._F
    except AttributeError:
      self.F()
    try:
      self._G
    except AttributeError:
      self.G()
    try:
      self._sum_occurance_matrix
    except AttributeError: 
      self.sum_occurance_matrix()
    try:
      self._sum_weighted_number
    except AttributeError: 
      self.sum_weighted_number()
    
    QsVSXS = (self._group_Q * self._sum_weighted_number).sum(axis=1)
    QsVSXS_sum_inv = 1.0/QsVSXS
    n2F = -2.0*self._F
    F2_2 = 2.0*self._F*self._F
    QsVSXS_sum_inv2 = 2.0*QsVSXS_sum_inv
    nffVSj = -self._F[:,np.newaxis]*self._sum_occurance_matrix[np.newaxis,:]
    n2FVsK = n2F[:,np.newaxis] * self._sum_occurance_matrix[np.newaxis,:]

    # for vec0 calculation
    # Reshape and expand arrays for broadcasting
    nffVSj_expanded = nffVSj[np.newaxis,:,:]

    # Transpose VSXS to align dimensions
    VSXS_transposed = np.transpose(self._sum_weighted_number[:, :, np.newaxis] , (1, 2, 0)).reshape(len(self._group_Q), np.shape(self.z)[0], 1)

    # Add vs and multiply by Qs
    vec0 = np.sum(self._group_Q[:,np.newaxis,np.newaxis] * (VSXS_transposed * nffVSj_expanded + self._occurance_matrix.reshape(len(self._group_Q),1,np.shape(self.z)[1])), axis=0)

    # for tot0 calculation
    # Reshape variables to align dimensions for broadcasting
    # Reshaping variables to add singleton dimensions where necessary
    # n2FVsK: (3, 1, 2, 1)
    n2FVsK_expanded = n2FVsK[:, np.newaxis, :, np.newaxis]  # (3, 1, 2, 1)
    # VSXS: (3, 1, 1, 4)
    VSXS_expanded = self._sum_weighted_number[:, np.newaxis, np.newaxis, :]      # (3, 1, 1, 4)
    # VS: (1, 2, 1, 1)
    VS_j_expanded = self._sum_occurance_matrix[np.newaxis, :, np.newaxis, np.newaxis]  # (1, N, 1, 1)
    # VS: (1, 1, 2, 1)
    VS_k_expanded = self._sum_occurance_matrix[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, N, 1)
    # self._occurance_matrix[n, k]: (1, 1, 2, 4)
    occurance_matrix_nk_expanded = self._occurance_matrix.T[np.newaxis, np.newaxis, :, :]  # vs.T shape is (2, 4), transpose vs to get (2, N_groups)
    # vs[n, j]: (1, 2, 1, 4)
    occurance_matrix_nj_expanded = self._occurance_matrix.T[np.newaxis, :, np.newaxis, :]  # vs.T shape is (2, 4)
    # Qs: (1, 1, 1, 4)
    Qs_expanded = self._group_Q[np.newaxis, np.newaxis, np.newaxis, :]  # (1, 1, 1, N_groups)

    # Compute the first term: VS[j] * (n2FVsK * VSXS + vs[n, k])
    term1 = VS_j_expanded * (n2FVsK_expanded * VSXS_expanded + occurance_matrix_nk_expanded)

    # Compute the second term: VS[k] * vs[n, j]
    term2 = VS_k_expanded * occurance_matrix_nj_expanded

    # Sum both terms
    terms = term1 + term2  # Shape: (3, 2, 2, 4)

    # Multiply by Qs[n]
    terms *= Qs_expanded  # Broadcasting over Qs

    # Sum over n (axis=-1) to get tot0 of shape (3, 2, 2)
    tot0 = np.sum(terms, axis=-1)

    # Multiply by F and QsVSXS_sum_inv
    F_expanded = self._F[:, np.newaxis, np.newaxis]  # (3, 1, 1)
    QsVSXS_sum_inv_expanded = QsVSXS_sum_inv[:, np.newaxis, np.newaxis]  # (3, 1, 1)

    tot0 *= F_expanded * QsVSXS_sum_inv_expanded # shape: (3, 2, 2)

    # Initialize d2Thetas_dxixjs with the appropriate shape
    d2Thetas_dxixjs = np.zeros((np.shape(self.z)[0], self.N, self.N, self.M))

    for f in range(np.shape(self.z)[0]):
      for j in range(self.N):
        VS_j = self._sum_occurance_matrix[j]  # Scalar
        vs_j = self._occurance_matrix[:, j]  # Shape: (self.M,)
        vec0_fj = vec0[f, j]  # Scalar

        # Shapes for broadcasting
        VS_k = self._sum_occurance_matrix  # Shape: (self.N,)
        vs_k = self._occurance_matrix  # Shape: (self.M, self.N)
        vec0_fk = vec0[f]  # Shape: (self.N,)

        # Compute terms using broadcasting
        term1 = -self._F[f] * (VS_j * vs_k + VS_k[np.newaxis, :] * vs_j[:, np.newaxis])  # Shape: (N_groups, N)
        term2 = self._sum_weighted_number[f, :, np.newaxis] * tot0[f, j, :]  # Shape: (N_groups, N)
        term3 = F2_2[f] * VS_j * VS_k[np.newaxis, :] * self._sum_weighted_number[f, :, np.newaxis]  # Shape: (N_groups, N)

        # Compute the temp term using broadcasting
        temp = QsVSXS_sum_inv[f] * (
            QsVSXS_sum_inv2[f] * self._sum_weighted_number[f, :, np.newaxis] * vec0_fj * vec0_fk[np.newaxis, :]
            - vs_j[:, np.newaxis] * vec0_fk[np.newaxis, :] - vs_k * vec0_fj
            + self._F[f] * self._sum_weighted_number[f, :, np.newaxis] * (VS_j * vec0_fk[np.newaxis, :] + VS_k[np.newaxis, :] * vec0_fj)
        )  # Shape: (N_groups, N)

        # Sum all terms
        v = term1 + term2 + term3 + temp  # Shape: (N_groups, N)

        # Compute the final result with broadcasting
        result = (v * self._group_Q[:, np.newaxis] * QsVSXS_sum_inv[f]).T  # Shape: (N, N_groups)

        # Assign the computed values to the result tensor
        d2Thetas_dxixjs[f, j, :, :] = result

    self._d2Thetas_dxixjs = d2Thetas_dxixjs
    return self._d2Thetas_dxixjs


  def d2GE_dxixjs(self):
    r"""
    Calculate the second composition derivative of excess Gibbs energy with respect to mol fractions of molecule :math:`j` and molecule :math:`k`.

    .. math::
        \frac{1}{RT}\frac{\partial^2 G^E}{\partial x_i \partial x_j} = 
          \frac{\partial \ln \gamma_i^c}{\partial x_j}
        + \frac{\partial \ln \gamma_i^r}{\partial x_j}
        + \frac{\partial \ln \gamma_j^c}{\partial x_i}
        + \frac{\partial \ln \gamma_j^r}{\partial x_i}
        +\sum_k^N \left(
          \frac{\partial^2 \ln \gamma_k^c}{\partial x_i \partial x_j}
          + \frac{\partial^2 \ln \gamma_k^r}{\partial x_i \partial x_j}
        \right)

    :return: second composition derivative of excess Gibbs energy
    :rtype: numpy.ndarray
    """
    try:
      self._dlngammas_c_dxs
    except AttributeError:
      self.dlngammas_c_dxs()
    try:
      self._dlngammas_r_dxs
    except AttributeError:
      self.dlngammas_r_dxs()
    try:
      self._d2lngammas_c_dxixjs
    except AttributeError:
      self.d2lngammas_c_dxixjs()
    try:
      self._d2lngammas_r_dxixjs
    except AttributeError:
      self.d2lngammas_r_dxixjs()

    # Sum the terms and their transposes over the axes
    dGE_initial = self._dlngammas_c_dxs + self._dlngammas_r_dxs + self._dlngammas_c_dxs.transpose(0, 2, 1) + self._dlngammas_r_dxs.transpose(0, 2, 1)

    # Expand the dimensions of z to align for broadcasting
    z_expanded = self.z[:, :, np.newaxis, np.newaxis]  

    # Sum over k (axis=1) after multiplying z with the sum of second derivatives
    sum_over_k = np.sum(z_expanded * (self._d2lngammas_c_dxixjs + self._d2lngammas_r_dxixjs), axis=1)

    # Combine all terms
    self._d2GE_dxixjs = self.Rc * self.T * (dGE_initial + sum_over_k)  # Shape: (3, 2, 2)
    return self._d2GE_dxixjs


  def dmu_dz(self):
    r"""
    Calculates the derivative of the chemical potential of molecule :math:`i` (:math:`\mu_i`) with respect to the mol fraction of molecule :math:`j`.

    .. math::
      \frac{1}{RT}\frac{\partial \mu_i}{\partial x_j} = 
        \frac{1}{x_i}
        + \frac{\partial \ln \gamma_i^c}{\partial x_j}
        + \frac{\partial \ln \gamma_i^r}{\partial x_j}
        + \frac{\partial \ln \gamma_j^c}{\partial x_i}
        + \frac{\partial \ln \gamma_j^r}{\partial x_i}
        +\sum_k \left(
          \frac{\partial^2 \ln \gamma_k^c}{\partial x_i \partial x_j}
          + \frac{\partial^2 \ln \gamma_k^r}{\partial x_i \partial x_j}
        \right)
       
    :return: derivative of chemical potential with respect to composition
    :rtype: numpy.ndarray
    """
    try:
      self._dlngammas_c_dxs
    except AttributeError:
      self.dlngammas_c_dxs()
    try:
      self._dlngammas_r_dxs
    except AttributeError:
      self.dlngammas_r_dxs()
    try:
      self._d2lngammas_c_dxixjs
    except AttributeError:
      self.d2lngammas_c_dxixjs()
    try:
      self._d2lngammas_r_dxixjs
    except AttributeError:
      self.d2lngammas_r_dxixjs()

    # Sum the terms and their transposes over the axes
    dmu_initial = self._dlngammas_c_dxs + self._dlngammas_r_dxs + self._dlngammas_c_dxs.transpose(0, 2, 1) + self._dlngammas_r_dxs.transpose(0, 2, 1) + 1/self.z[:,np.newaxis]

    # Expand the dimensions of z to align for broadcasting
    z_expanded = self.z[:, :, np.newaxis, np.newaxis]  

    # Sum over k (axis=1) after multiplying z with the sum of second derivatives
    sum_over_k = np.sum(z_expanded * (self._d2lngammas_c_dxixjs + self._d2lngammas_r_dxixjs), axis=1)

    # Combine all terms
    self._dmu_dz = self.Rc * self.T * (dmu_initial + sum_over_k)  # Shape: (3, 2, 2)
    return self._dmu_dz
  
  def Mij(self):
    r"""
    Calculate the :math:`M` matrix with elements :math:`M_{i,j}` for molecules :math:`i` and :math:`j` from :func:`dmu_dz`.

    .. math::
      M_{i,j} = 
      \begin{align}
      \begin{cases}
        & \frac{\partial \mu_i}{\partial x_j}, & \text{if } i = j \\
        & \frac{\partial \mu_i}{\partial x_j} - \frac{\partial \mu_i}{\partial x_{N-1}} - \frac{\partial \mu_{N-1}}{\partial x_j} + \frac{\partial \mu_{N-1}}{\partial x_{N-1}}, & \text{if } i \neq j \\
      \end{cases}
      \end{align}

    :return: :math:`M` matrix from second derivative of Gibbs mixing free energy with respect to mol fractions
    :rtype: numpy.ndarray
    """
    try:
        self._dmu_dz
    except AttributeError:
        self.dmu_dz()

    Mij = np.empty((np.shape(self.z)[0], np.shape(self.z)[1], np.shape(self.z)[1]))
    n = np.shape(self.z)[1]

    for i in range(n):
      for j in range(n):
        if i == j:
          Mij[:,i,j] = self._dmu_dz[:,i,j]
        else:
          Mij[:,i,j] = self._dmu_dz[:,i,j] - self._dmu_dz[:,i,n-1] - self._dmu_dz[:,n-1,j] + self._dmu_dz[:,n-1,n-1]

    self._Mij = Mij
    return self._Mij

  def Hij(self):
    r"""
    Calculate the Hessian (:math:`H`) of Gibbs mixing free energy, with elements :math:`H_{ij}` for molecules :math:`i` and :math:`j`.

    .. math::
      H_{i,j} = M_{i,j} - M_{i,N-1} - M_{N-1,j} + M_{N-1,N-1}

    :return: Hessian matrix
    :rtype: numpy.ndarray    
    """
    try:
        self._Mij
    except AttributeError:
        self.Mij()

    Hij = np.empty((np.shape(self.z)[0], np.shape(self.z)[1] - 1, np.shape(self.z)[1] - 1))
    n = np.shape(self.z)[1] - 1
    for i in range(n):
      for j in range(n):
        Hij[:,i,j] = self._Mij[:,i,j] - self._Mij[:,i,n] - self._Mij[:,n,j] + self._Mij[:,n,n]

    self._Hij = Hij
    return self._Hij

  
  def det_Hij(self):
    r"""
    Calculates the determinant (:math:`|H|`) of Hessian matrix of Gibbs mixing free energy.

    :return: Hessian determinant
    :rtype: numpy.ndarray
    """
    try:
      self._Hij 
    except AttributeError:
      self.Hij()

    return np.linalg.det(self._Hij)
  
