import numpy as np
import pandas as pd
from pathlib import Path
import math, os, sys
from scipy.interpolate import interp1d
import scipy.optimize as sco
from scipy import constants
from copy import copy
from rdkit import Chem
from rdkit.Chem import AllChem

from .models import FH, NRTL, UNIQUAC, UNIFAC, QuarticModel
from .models.unifac import get_unifac_version
from .get_molecular_properties import load_molecular_properties, search_molecule
from .models.cem import CEM
from .kbi import mkdr, KBI
from .functions import get_solute_molid, mol2vol

def spinodal_fn(z, Hij):
  r"""
  Calculates mole fractions where the Hessian of the Gibbs mixing free energy is zero, indicating spinodal points.

  :param z: mol fraction array with shape ``(number of compositions, number of components)``.
  :type z: numpy.ndarray
  :param Hij: determinant of Hessian matrix of the Gibbs mixing free energy.
  :type Hij: numpy.ndarray
  :return: mol fractions of the spinodal curve. Returns an array of shape ``(2, number of components)`` for binary mixtures or ``None`` if fewer or more than two spinodal points are detected (excluding the first and last 5 points).
  :rtype: numpy.ndarray or None
  """
  sign_changes = np.diff(np.sign(Hij))  # diff of signs between consecutive elements
  spin_inds = [s for s, sc in enumerate(sign_changes) if sc != 0 and ~np.isnan([sc]) if (s not in range(5)) and (s not in range(len(sign_changes)-5,len(sign_changes)))]
  if len(spin_inds) == 2:
    if z.shape[1] > 1:
      return z[spin_inds,:]
    else:
      return z[spin_inds]
  else:
    return None
  
def binodal_fn(num_comp, rec_steps, G_mix, activity_coefs, solute_idx=None, bi_min=-np.inf, bi_max=np.inf):
  r"""
  Calculates mole fractions of the coexistence curve (binodal) using the convex envelope method (:class:`picmol.models.cem.CEM`), which applies the isoactivity criterion.

  :param num_comp: number of components in the mixture.
  :type num_comp: int
  :param rec_steps: spacing for the discretization of the composition space.
  :type rec_steps: int
  :param G_mix: Gibbs mixing free energy array.
  :type G_mix: numpy.ndarray
  :param activity_coefs: array of activity coefficients for each component at each composition in the discretized space.
  :type activity_coefs: numpy.ndarray
  :param solute_idx: index of the solute component for reporting mole fractions (only relevant for binary systems).
  :type solute_idx: int, optional
  :param bi_min: lower bound for mole fractions to be considered for binodals. Defaults to negative infinity.
  :type bi_min: float, optional
  :param bi_max: upper bound for mole fractions to be considered for binodals. Defaults to positive infinity.
  :type bi_max: float, optional
  :return: for binary systems, returns a tuple of two floats representing the minimum and maximum binodal mole fractions of the solute. Returns an array for multicomponent systems. Returns ``(np.nan, np.nan)`` if no valid binodal points are found for binary systems.
  :rtype: tuple[float, float] or numpy.ndarray or tuple[np.nan, np.nan]
  """
  bi_obj = CEM(num_comp=num_comp, rec_steps=rec_steps, G_mix=G_mix, activity_coefs=activity_coefs)
  bi_vals = bi_obj.binodal_matrix_molfrac
  if num_comp < 3:
    bi_vals = bi_vals[:,:,solute_idx]
    if len(bi_vals) > 0:
      mask = np.all((bi_vals > bi_min) & (bi_vals < bi_max), axis=1)
      bi_filter = bi_vals[mask]
      try:
        bi1, bi2 = bi_filter.min(), bi_filter.max()
        return bi1, bi2
      except:
        return np.nan, np.nan
  else:
    return bi_vals
  return np.nan, np.nan

def numerical_binodal_fn(x, GM, dGM, sp1=None, sp2=None):
  # numerical binodal function that searches for binodal points that minimize a function for binary mixture
  # if both spinodals are np.nan, don't execute binodal function
  if np.isnan(sp1) and np.isnan(sp2):
    return {}

  if np.all([sp1, sp2] == None):
    dGM = np.gradient(GM, x)
    d2GM = np.gradient(dGM, x)
    sps = spinodal_fn(x, d2GM)

  iiall = ~np.isnan(dGM) & ~np.isnan(GM)
  ii1 = (x < sp1) & ~np.isnan(dGM) & ~np.isnan(GM)
  ii2 = (x > sp2) & ~np.isnan(dGM) & ~np.isnan(GM)

  Gx_from_x = interp1d(x[iiall], GM[iiall], kind='cubic', fill_value="extrapolate")
  x2_from_dGx = interp1d(dGM[ii2], x[ii2])

  def objective(vars):
    '''calculates binodal pts for a binary mixture'''
    global x1, x2, dGx, dGx2
    x1, x2 = vars
    # mixing free energy
    Gx = Gx_from_x(x1)
    Gx2 = Gx_from_x(x2)
    
    # Derivatives
    h = 1e-5
    dGx = (Gx_from_x(x1 + h) - Gx_from_x(x1 - h)) / (2 * h)
    dGx2 = (Gx_from_x(x2 + h) - Gx_from_x(x2 - h)) / (2 * h)

    # binodal definitions
    eq1 = dGx2 - dGx
    eq2 = dGx - (Gx2 - Gx)/(x2 - x1)
    eq3 = dGx2 - (Gx2 - Gx)/(x2 - x1)  
    
    # return sum of eqns
    return eq1 + eq2 + eq3
  
  # minimize the binodal funciton to solve for binodal pts
  bis = sco.minimize(objective, x0=[0.1, 0.99], bounds=((1E-5,sp1),(sp2,0.9999)))

  bi1 = bis.x[0]
  bi2 = np.interp(dGx, dGM[ii2],  x[ii2])
  return bi1, bi2

def calculate_saxs_Io(z, molar_vol, n_electrons, Hij, T):
  r"""
  Estimate the small-angle X-ray scattering (SAXS) zero-angle intensity, :math:`I_0` (cm\ :sup:`-1`), as a function of temperature and composition.

  The zero-angle scattering intensity is calculated using the formula:

  .. math::
    I_0 = r_e^2 \left(\frac{\partial\rho}{\partial x}\right)^2 S_0

  where the electron density contrast, :math:`\frac{\partial \rho}{\partial x}`, is given by:

  .. math::
    \frac{\partial \rho}{\partial x} = N_A\sum_{i=1}^{N-1} \frac{N_i^e - N_n^e}{\overline{V}} - \overline{N^e}\left(\frac{V_i - V_N}{\overline{V^2}}\right)

  and the structure factor at zero angle, :math:`S_0`, is:

  .. math::
    S_0 = \frac{k_b T\overline{V}}{|H^x|}

  The average molar volume, :math:`\overline{V}`, and average number of electrons, :math:`\overline{N^e}`, are calculated as:

  .. math::
    \overline{V} = \sum_i^n x_i V_i

  .. math::
    \overline{N^e} = \sum_i^n x_i N_i^e

  where:
  
    * :math:`r_e` is the electron radius.
    * :math:`N_A` is Avogadro's number.
    * :math:`N_i^e` is the number of electrons in component :math:`i`.
    * :math:`V_i` is the molar volume of component :math:`i`.
    * :math:`\overline{V}` is the average molar volume of the mixture.
    * :math:`\overline{N^e}` is the average number of electrons in the mixture.
    * :math:`|H^x|` is the determinant of the Gibbs mixing free energy Hessian.
    * :math:`k_b` is the Boltzmann constant.
    * :math:`T` is the temperature.

  :param z: mol fraction array with shape ``(number of compositions, number of components)``.
  :type z: numpy.ndarray
  :param molar_vol: array of molar volumes for each component.
  :type molar_vol: numpy.ndarray
  :param n_electrons: array of the number of electrons for each component.
  :type n_electrons: numpy.ndarray
  :param Hij: Hessian matrix of the Gibbs mixing free energy.
  :type Hij: numpy.ndarray
  :param T: temperature (K)
  :type T: float
  :return: array of the zero-angle SAXS intensity, :math:`I_0`, for each composition.
  :rtype: numpy.ndarray
  """
  re2 = 7.9524E-26 # cm^2
  R_kJ = constants.R / 1000 # kJ/mol
  kb = R_kJ / constants.N_A
  num_comp = z.shape[1]
  V_bar_i0 = z @ molar_vol
  N_bar = z @ n_electrons
  rho_e = 1/V_bar_i0

  drho_dx = np.zeros(z.shape[0])
  for j in range(num_comp-1):
    drho_dx += constants.N_A*(rho_e*(n_electrons[j]-n_electrons[-1]) - N_bar*rho_e*(molar_vol[j]-molar_vol[-1])/V_bar_i0)

  S0 = kb * T * V_bar_i0 / Hij 
  I0 = re2 * (drho_dx**2) * S0 # cm^-1
  return I0


def Tc_search(smiles: list, lle_type=None, Tmin=100, Tmax=500, dT=5, unif_version="unifac"):
  r"""
  Calculates the critical temperature (K) for a binary system using the UNIFAC model. The critical temperature is identified by finding the temperature at which spinodal points appear (Hessian of Gibbs mixing free energy becomes zero). The calculation can be optimized by specifying the type of LLE.

  :param smiles: list of SMILES representations of the two molecules in the binary system.
  :type smiles: list
  :param lle_type: type of liquid-liquid equilibrium transition. Options are:
      
      * 'ucst': Upper critical solution temperature (critical temperature is the maximum on the phase diagram).
      * 'lcst': Lower critical solution temperature (critical temperature is the minimum on the phase diagram).
  
  :type lle_type: str, optional
  :param Tmin: minimum temperature (K) for the search. Required for ``lle_type='lcst'`` or ``None``. Defaults to 100 K.
  :type Tmin: float, optional
  :param Tmax: maximum temperature (K) for the search. Required for ``lle_type='ucst'`` or ``None``. Defaults to 500 K.
  :type Tmax: float, optional
  :param dT: temperature step (K) for the search when ``lle_type`` is ``None``. Defaults to 5 K.
  :type dT: float, optional
  :param unif_version: version of the UNIFAC model to use. Options are:
     
      * 'unifac': Original UNIFAC.
      * 'unifac-lle': Updated parameters for LLE.
      * 'unifac-il': Includes support for ionic liquid molecules.
  
  :type unif_version: str, optional
  :return: The critical temperature (K). Returns ``np.nan`` if the critical temperature cannot be found.
  :rtype: float or np.nan
  """
  if "md" in unif_version or 'kbi' in unif_version:
    version = "unifac-kbi"
  elif "il" in unif_version:
    version = "unifac-il"
  else:
    version = "unifac"

  i=0
  if lle_type is not None:
    T_initial = {"ucst": Tmax, "lcst": Tmin}
    T = T_initial[lle_type]
    dT = 50
    while T >= 0 and T <= 1200:
      model = UNIFAC(T=T, smiles=smiles, version=version)
      d2GM = model.det_Hij()
      sign_changes = np.diff(np.sign(d2GM))  # diff of signs between consecutive elements
      spin_inds = [s for s, sc in enumerate(sign_changes) if sc != 0 and ~np.isnan([sc])]
      if len(spin_inds) > 0 and dT == 50:
        if T == Tmax and i==0:
          T += 50
          i+=1
          continue
        if lle_type.lower() == "ucst":
          T += dT-20
        else:
          T -= dT-20
        dT = 20
      elif len(spin_inds) > 0 and dT == 20:
        if lle_type.lower() == "ucst":
          T += dT-5
        else:
          T -= dT-5
        dT = 5
      elif len(spin_inds) > 0 and dT == 5:
        return T
      else:
        if lle_type.lower() == "ucst":
          T -= dT
        else:
          T += dT
    return np.nan
  # if lle type is not known, calculate spinodals for all T
  # Tc is where there is the smallest difference between spinodal values
  else:
    T_values = np.arange(Tmin, Tmax+1, dT)
    sp_matrix = np.full((len(T_values), len(smiles)), fill_value=np.nan)
    for t, T in enumerate(T_values):
      model = UNIFAC(T=T, smiles=smiles, version=version)
      d2GM = model.det_Hij()
      sp = spinodal_fn(model.z, d2GM)
      if sp is not None:      
        sp_matrix[t] = [sp[:,0].min(), sp[:,0].max()]
      else:
        sp_matrix[t] = [np.nan, np.nan]
    # get Tc where there is the smallest difference between spinodal values
    try:
      nan_mask = ~np.isnan(sp_matrix[:,0]) & ~np.isnan(sp_matrix[:,1])
      sp_matrix_filter = sp_matrix[nan_mask]
      T_values_filter = T_values[nan_mask]
      crit_ind = np.abs(sp_matrix_filter[:,0]-sp_matrix_filter[:,1]).argmin()
      return T_values_filter[crit_ind]
    except:
      return np.nan

class ThermoModel:
  r"""
  Calculates Gibbs mixing free energy and related mixing properties as a function of temperature for LLE analysis.

  This class provides a framework for applying various thermodynamic models to analyze
  LLE in mixtures. It supports both binary and multicomponent
  systems, depending on the chosen thermodynamic model. The class initializes based on a
  selected model name, a KBI (Kirkwood-Buff Integrals) model object, and a temperature
  range for analysis.

  :param model_name: thermodynamic model to implement. Available options include:

      * 'quartic': Numerical model using a 4th order Taylor series expansion (:class:`picmol.models.numerical.QuarticModel`). Supported for multicomponent mixtures.
      * 'uniquac': UNIQUAC (Universal Quasi-Chemical) thermodynamic model (:class:`picmol.models.uniquac.UNIQUAC`). Supported for multicomponent mixtures.
      * 'unifac': UNIFAC (Universal Functional-group Activity Coefficients) thermodynamic model (:class:`picmol.models.unifac.UNIFAC`). Supported for multicomponent mixtures.
      * 'fh': Flory-Huggins thermodynamic model (:class:`picmol.models.fh.FH`). Supported for binary systems only.
      * 'nrtl': NRTL (Non-Random Two-Liquid) thermodynamic model (:class:`picmol.models.nrtl.NRTL`). Supported for binary systems only.

  :type model_name: str
  :param KBIModel: KBI class object containing Kirkwood-Buff integrals and related data.
                    This object provides information about the mixture components and their interactions.
  :type KBIModel: KBI
  :param Tmin: minimum temperature (K) for the temperature scaling analysis. Defaults to 100 K.
  :type Tmin: float, optional
  :param Tmax: maximum temperature (K) for the temperature scaling analysis. Defaults to 400 K.
  :type Tmax: float, optional
  :param dT: temperature step (K) for the temperature scaling analysis. Defaults to 10 K.
  :type dT: float, optional
  """
  """run thermo model on kbi results"""
  def __init__(
      self, 
      model_name: str, # thermo model
      KBIModel, # option to feed in kbi model
      Tmin=100, Tmax=400, dT=10, # temperature range
    ):

    self.kbi_model = KBIModel
    self.identifiers = self.kbi_model.unique_mols
    self.identifier_type = "mol_id"
    self.model_name = model_name.lower()
    self.save_dir = mkdr(f"{self.kbi_model.kbi_method_dir}/{self.model_name}/")

    # grab solute from KBIModel
    self.kbi_model.solute_mol = self.kbi_model.solute
    
    # get interaction parameters
    self.IP = None
    if self.model_name in ["nrtl", "uniquac"]:
      # get IP parameters for model type
      IP_map = {
        "nrtl": self.kbi_model.nrtl_taus, 
        "uniquac": self.kbi_model.uniquac_du, 
      }
      self.IP = IP_map[self.model_name]

    # initialize temperatures
    self.Tmin = Tmin
    self.Tmax = Tmax
    self.dT = dT

    # get type of thermo model
    model_map = {
      "fh": FH, 
      "nrtl": NRTL, 
      "uniquac": UNIQUAC, 
      "unifac": UNIFAC, 
      "quartic": QuarticModel, 
    }
    self.model_type = model_map[self.model_name]

    ### initialize thermodynamic model
    # for unifac model
    if self.model_type == UNIFAC:
      self.unif_version = get_unifac_version('unifac', self.kbi_model.smiles)
      self.model = self.model_type(T=Tmax, smiles=self.kbi_model.smiles, version=self.unif_version)
    # quartic/numerical
    elif self.model_type == QuarticModel:
      self.model = self.model_type(z_data=KBIModel.z, Hmix=KBIModel.Hmix(), Sex=KBIModel.SE(), molar_vol=self.kbi_model.molar_vol, gid_type='vol')
    # Flory-Huggins
    elif self.model_type == FH:
      self.model = self.model_type(smiles=self.kbi_model.smiles, phi=KBIModel.v[:,self.kbi_model.solute_loc], Smix=KBIModel.SM(), Hmix=KBIModel.Hmix())
    # NRTL 
    elif self.model_type == NRTL:
      self.model = self.model_type(IP=self.IP)
    # UNIQUAC
    elif self.model_type == UNIQUAC:
      self.model = self.model_type(smiles=self.kbi_model.smiles, IP=self.IP)

    # get volume fraction of composition
    self.z = self.model.z
    self.v = mol2vol(self.z, self.kbi_model.molar_vol)

  # add more detailed doc to this!!!
  def run(self):
    r"""
    Performs temperature scaling analysis to estimate Gibbs mixing energy, phase diagram,
    and calculate SAXS I\ :sub:`0` values.

    This method iterates over the defined temperature range to calculate thermodynamic
    properties relevant for LLE. It distinguishes between binary
    and multicomponent systems to apply appropriate calculation methods.

    **Attributes (after running this method):**

    * ``GM`` (kJ/mol): Gibbs mixing energy as a function of temperature and composition.
    * ``det_Hij``: determinant of Hessian of Gibbs mixing energy with respect to composition.
    * ``x_sp``: mol fractions at spinodal compositions as a function of temperature.
    * ``x_bi``: mol fractions at binodal compositions as a function of temperature.
    * ``v_sp``: volume fractions at spinodal compositions as a function of temperature.
    * ``v_bi``: volume fractions at binodal compositions as a function of temperature.
    * ``GM_sp`` (kJ/mol, binary only): Gibbs mixing energy at spinodal compositions.
    * ``GM_bi`` (kJ/mol, binary only): Gibbs mixing energy at binodal compositions.
    * ``I0_arr``: Small-Angle X-ray Scattering (SAXS) zero-angle scattering intensity
      as a function of temperature and composition.
    * ``x_I0_max``: Widom line compositions in mole fraction as a function of temperature.
    * ``v_I0_max``: Widom line compositions in volume fraction as a function of temperature.

    The calculated results are stored as attributes of the :class:`ThermoModel` object.
    """
    if self.kbi_model.num_comp == 2:
      self._binary_temperature_scaling()
    else:
      self._multicomp_temperature_scaling()

  @property
  def T_values(self):
    r""":return: np.array of temperature values (K) for scaling
    :rtype: numpy.ndarray
    """
    return np.arange(self.Tmin, self.Tmax+1E-3, self.dT)[::-1]

  def _binary_temperature_scaling(self):

    self.GM = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
    self.det_Hij = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
    self.x_sp = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
    self.x_bi = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
    self.GM_sp = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
    self.GM_bi = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
    self.v_sp = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
    self.v_bi = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
    self.I0_arr = np.empty((len(self.T_values), self.z.shape[0]))
    self.x_I0_max = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
    self.v_I0_max = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)

    
    for t, T in enumerate(self.T_values):
      # set variables to nan
      sp1, sp2, bi1, bi2 = np.empty(4)*np.nan

      # for just FH model
      if self.model_type == FH:
        self.GM[t,:] = self.model.GM(self.v[:,0], T)
        self.det_Hij[t,:] = self.model.det_Hij(self.v[:,0],T)
        try:
          sp1, sp2, bi1, bi2 = self.model.calculate_spindoals_binodals(self.model.phic, T)
        except:
          pass
      
      # for all other thermodynamic models
      else:
        # unifac, requires a new model object at each temperature
        if self.model_type == UNIFAC:
          self.model = self.model_type(T=T, smiles=self.kbi_model.smiles, version=self.unif_version)
          gm = self.model.GM()
          dgm = self.model.dGM_dxs().flatten()
          self.GM[t,:] = gm
          self.det_Hij[t,:] = self.model.det_Hij()
          gammas = self.model.gammas()

        # other thermo models (uniquac, nrtl, quartic-numerical)
        else: 
          gm = self.model.GM(T)
          dgm = self.model.dGM_dxs(T)[:,self.kbi_model.solute_loc]
          self.GM[t,:] = gm
          self.det_Hij[t,:] = self.model.det_Hij(T)
          gammas = self.model.gammas(T)

        # now get spinodals and binodals
        sp_vals = spinodal_fn(self.z, self.det_Hij[t,:])
        if sp_vals is not None:    
          sp_vals = sp_vals[:,self.kbi_model.solute_loc]  
          sp1, sp2 = sp_vals.min(), sp_vals.max()
          # quartic doesn't have gammas --> uses the numerical binodal function
          if self.model_type in [NRTL]:
            bi1, bi2 = numerical_binodal_fn(x=self.z[:,self.kbi_model.solute_loc], sp1=sp1, sp2=sp2, GM=gm, dGM=dgm)
          # thermodynamic models can use convex envelope method, which requires Gmix and activity coefficients
          else:
            bi1, bi2 = binodal_fn(num_comp=self.kbi_model.num_comp, rec_steps=self.model.rec_steps, G_mix=gm, activity_coefs=gammas, solute_idx=self.kbi_model.solute_loc, bi_min=0.001, bi_max=0.99)
        
      self.x_sp[t,:] = [sp1, sp2]
      self.x_bi[t,:] = [bi1, bi2]

      if np.all(~np.isnan([sp1, sp2])):
        sp1_ind = np.abs(self.z[:,self.kbi_model.solute_loc] - sp1).argmin()
        sp2_ind = np.abs(self.z[:,self.kbi_model.solute_loc] - sp2).argmin()
        self.v_sp[t,:] = [self.v[sp1_ind, self.kbi_model.solute_loc], self.v[sp2_ind, self.kbi_model.solute_loc]]
        self.GM_sp[t,:] = [self.GM[t,sp1_ind], self.GM[t,sp2_ind]]
      
      if np.all(~np.isnan([bi1, bi2])):
        bi1_ind = np.abs(self.z[:,self.kbi_model.solute_loc] - bi1).argmin()
        bi2_ind = np.abs(self.z[:,self.kbi_model.solute_loc] - bi2).argmin()
        self.v_bi[t,:] = [self.v[bi1_ind, self.kbi_model.solute_loc], self.v[bi2_ind, self.kbi_model.solute_loc]]
        self.GM_bi[t,:] = [self.GM[t,bi1_ind], self.GM[t,bi2_ind]]
        
      # calculate I0
      self.I0_arr[t] = calculate_saxs_I0(z=self.z, molar_vol=self.kbi_model.molar_vol, n_electrons=self.kbi_model.n_electrons, Hij=self.det_Hij[t], T=T)
      # get widom line where spinodals are not found, i.e., where mixture is stable
      if np.all(np.isnan(self.x_sp[t])):
        I0_mask = ~np.isnan(self.I0_arr[t])
        if len(I0_mask) > 0:
          # get I0 max
          I0_max = max(self.I0_arr[t][I0_mask])
          # get widom line in mol frac and vol frac
          self.v_I0_max[t] = self.v[I0_mask][np.where(self.I0_arr[t][I0_mask]==I0_max)[0][0]]
          self.x_I0_max[t] = self.z[I0_mask][np.where(self.I0_arr[t][I0_mask]==I0_max)[0][0]]

    # get critical point
    # find where there is the smallest difference between the spinodal values
    nan_mask = ~np.isnan(self.x_sp[:,0]) & ~np.isnan(self.x_sp[:,1])
    x_filter = self.x_sp[nan_mask]
    T_values_filter = self.T_values[nan_mask]
    crit_ind = np.abs(x_filter[:,0]-x_filter[:,1]).argmin()
    if self.model_type == FH:
      self.xc = self.model.xc
      self.phic = self.model.phic
    else:
      self.xc = np.mean(x_filter[crit_ind,:])
      self.phic = mol2vol(([self.xc, 1-self.xc]), self.kbi_model.molar_vol)
    self.Tc = T_values_filter[crit_ind]


  def _multicomp_temperature_scaling(self):

    self.GM = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
    self.det_Hij = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
    self.x_sp = []
    self.x_bi = []
    self.v_sp = []
    self.v_bi = []
    self.I0_arr = np.empty((len(self.T_values), self.z.shape[0]))
    self.x_I0_max = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
    self.v_I0_max = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)

    for t, T in enumerate(self.T_values):
      if self.model_name == "unifac":
        self.model = UNIFAC(T=T, smiles=self.kbi_model.smiles, version=self.unif_version)
        GM = self.model.GM()
        det_Hij = self.model.det_Hij()
        gammas = self.model.gammas()
      elif self.model_name in ["uniquac", "quartic"]:
        GM = self.model.GM(T)
        det_Hij = self.model.det_Hij(T)
        gammas = self.model.gammas(T)

      self.GM[t] = GM

      if self.model_type != "unifac":
        self.det_Hij[t] = det_Hij
        # filter 2nd derivative for spinodal determination
        mask = (det_Hij > -1) & (det_Hij <= 1)
        sps = spinodal_fn(z=self.model.z, Hij=det_Hij)
        self.x_sp += [sps]
        self.v_sp += [mol2vol(sps, self.molar_vol)]

      bi_vals = binodal_fn(num_comp=self.kbi_model.num_comp, rec_steps=self.model.rec_steps, G_mix=GM, activity_coefs=gammas, solute_idx=self.kbi_model.solute_loc, bi_min=0.001, bi_max=0.99)
      self.x_bi += [bi_vals]
      self.v_bi += [mol2vol(bi_vals, self.molar_vol)]

      # calculate I0
      self.I0_arr[t] = calculate_saxs_Io(z=self.z, molar_vol=self.kbi_model.molar_vol, n_electrons=self.kbi_model.n_electrons, Hij=self.det_Hij[t], T=T)
      # get widom line where spinodals are not found, i.e., where mixture is stable
      if np.all(np.isnan(self.x_sp[t])):
        I0_mask = ~np.isnan(self.I0_arr[t])
        if len(I0_mask) > 0:
          # get I0 max
          I0_max = max(self.I0_arr[t][I0_mask])
          # get widom line in mol frac and vol frac
          self.v_I0_max[t] = self.v[I0_mask][np.where(self.I0_arr[t][I0_mask]==I0_max)[0][0]]
          self.x_I0_max[t] = self.z[I0_mask][np.where(self.I0_arr[t][I0_mask]==I0_max)[0][0]]


class UNIFACThermoModel:
  r"""
  Performs temperature scaling for LLE analysis using the UNIFAC model.

  This class is designed for purely predictive thermodynamic modeling based solely on the SMILES
  strings of the mixture components. It calculates Gibbs mixing free energy and related
  mixing properties as a function of temperature.

  :param smiles: list of SMILES strings for each component in the mixture.
  :type smiles: list
  :param mol_names: list of names for each component in the mixture.
  :type mol_names: list
  :param solute_idx: index of the solute component in the ``smiles`` and ``mol_names`` lists. Defaults to 0.
  :type solute_idx: int, optional
  :param Tmin: minimum temperature (K) for the temperature scaling analysis. Defaults to 100 K.
  :type Tmin: float, optional
  :param Tmax: maximum temperature (K) for the temperature scaling analysis. Defaults to 400 K.
  :type Tmax: float, optional
  :param dT: temperature step (K) for the temperature scaling analysis. Defaults to 10 K.
  :type dT: float, optional
  :param save_dir: directory to save output files (not currently used in the provided code). Defaults to None.
  :type save_dir: str, optional
  :param unif_version: UNIFAC version to use (e.g., 'unifac', 'unifac-lle', 'unifac-il'). Defaults to 'unifac'.
  :type unif_version: str, optional
  """
  def __init__(
      self, 
      smiles: list,
      mol_names: list,
      solute_idx=0,
      Tmin=100, Tmax=400, dT=10,
      save_dir=None,
      unif_version="unifac",
    ):

    self.save_dir = save_dir
    self.smiles = smiles
    self.solute_loc = solute_idx
    self.mol_name = mol_names
    self.solute_name = self.mol_name[self.solute_loc]
    self.model_name = "unifac"

    # initialize temperatures
    self.Tmin = Tmin
    self.Tmax = Tmax
    self.dT = dT

    ### initialize thermodynamic model
    # for unifac model
    self.unif_version = get_unifac_version(unif_version, self.smiles)
    self.model = UNIFAC(T=Tmax, smiles=self.smiles, version=self.unif_version)
    
    # get volume fraction of composition
    self.z = self.model.z
    self.v = mol2vol(self.z, self.molar_vol)


  def run(self):
    r"""
    Performs temperature scaling analysis using the UNIFAC model.

    This method calculates Gibbs mixing energy, spinodal and binodal compositions as a
    function of temperature. It distinguishes between binary and multicomponent systems
    to apply the appropriate calculation methods. The results are stored as attributes
    of the :class:`UNIFACThermoModel` object.
    """
    if self.num_comp == 2:
      self._binary_temperature_scaling()
    else:
      self._multicomp_temperature_scaling()

  @property
  def num_comp(self):
    r""":return: number of components in the mixture.
    :rtype: int
    """
    return len(self.smiles)

  @property
  def T_values(self):
    r""":return: array of temperature values (K) used for scaling.
    :rtype: numpy.ndarray"""
    return np.arange(self.Tmin, self.Tmax+1E-3, self.dT)[::-1]

  def compute_rdkit_properties(self):
    r"""
    Computes molecular properties using RDKit.

    This method calculates molecular weight, density, molar volume, number of electrons,
    and molecular charge for each component based on their SMILES strings.
    
    :return: dicstionary of molecular properties by molecule
    :rtype: dict
    """
    props = {
      'mol_wt': np.zeros(self.num_comp),
      'density': np.zeros(self.num_comp),
      'molar_vol': np.zeros(self.num_comp),
      'n_electrons': np.zeros(self.num_comp),
      'mol_charge': np.zeros(self.num_comp),
    }

    def get_electron_number(mol):
      atomic_numbers = []
      for atom in mol.GetAtoms():
        atomic_numbers += [atom.GetAtomicNum()]
      return sum(atomic_numbers)

    for i, smile in enumerate(self.smiles):
      mol_obj = Chem.MolFromSmiles(smile)
      mol_obj = AllChem.AddHs(mol_obj)
      props['n_electrons'][i] = get_electron_number(mol_obj)
      props['mol_wt'][i] = Chem.Descriptors.MolWt(mol_obj)
      props['mol_charge'][i] = Chem.GetFormalCharge(mol_obj)
      AllChem.EmbedMolecule(mol_obj, useRandomCoords=True)
      props['molar_vol'][i] = AllChem.ComputeMolVolume(mol_obj)
      props['density'][i] = props['mol_wt'][i]/props['molar_vol'][i] 
    
    self._rdkit_dict = props
    return self._rdkit_dict

  @property
  def mol_wt(self):
    r""":return: molar masses of the components from :func:`compute_rdkit_properties()` function.
    :rtype: numpy.ndarray"""
    try:
      self._rdkit_dict
    except AttributeError:
      self.compute_rdkit_properties()
    return self._rdkit_dict['mol_wt']
  
  @property
  def density(self):
    r""":return: densities of the components from :func:`compute_rdkit_properties()` function.
    :rtype: numpy.ndarray"""
    try:
      self._rdkit_dict
    except AttributeError:
      self.compute_rdkit_properties()
    return self._rdkit_dict['density']

  @property
  def molar_vol(self):
    r""":return: molar volumes of the components from :func:`compute_rdkit_properties()` function.
    :rtype: numpy.ndarray"""
    try:
      self._rdkit_dict
    except AttributeError:
      self.compute_rdkit_properties()
    return self._rdkit_dict['molar_vol']

  @property
  def n_electrons(self):
    r""":return: number of electrons for each component from :func:`compute_rdkit_properties()` function.
    :rtype: numpy.ndarray"""
    try:
      self._rdkit_dict
    except AttributeError:
      self.compute_rdkit_properties()
    return self._rdkit_dict['n_electrons']

  @property
  def mol_charge(self):
    r""":return: molecular charges of the components from :func:`compute_rdkit_properties()` function.
    :rtype: numpy.ndarray"""
    try:
      self._rdkit_dict
    except AttributeError:
      self.compute_rdkit_properties()
    return self._rdkit_dict['mol_charge']    
  
  def _binary_temperature_scaling(self):

    self.GM = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
    self.det_Hij = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
    self.x_sp = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
    self.x_bi = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
    self.GM_sp = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
    self.GM_bi = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
    self.v_sp = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
    self.v_bi = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
    self.I0_arr = np.empty((len(self.T_values), self.z.shape[0]))
    self.x_I0_max = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
    self.v_I0_max = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)

    for t, T in enumerate(self.T_values):
      # set variables to nan
      sp1, sp2, bi1, bi2 = np.empty(4)*np.nan

      # unifac, requires a new model object at each temperature
      self.model = UNIFAC(T=T, smiles=self.smiles, version=self.unif_version)
      gm = self.model.GM()
      self.GM[t,:] = gm
      self.det_Hij[t,:] = self.model.det_Hij()
      gammas = self.model.gammas()

      # now get spinodals and binodals
      sp_vals = spinodal_fn(self.z, self.det_Hij[t,:])
      if sp_vals is not None:    
        sp_vals = sp_vals[:,self.solute_loc]  
        sp1, sp2 = sp_vals.min(), sp_vals.max()
        bi1, bi2 = binodal_fn(num_comp=self.num_comp, rec_steps=self.model.rec_steps, G_mix=gm, activity_coefs=gammas, solute_idx=self.solute_loc, bi_min=0.001, bi_max=0.99)

      self.x_sp[t,:] = [sp1, sp2]
      self.x_bi[t,:] = [bi1, bi2]

      if np.all(~np.isnan([sp1, sp2])):
        sp1_ind = np.abs(self.z[:,self.solute_loc] - sp1).argmin()
        sp2_ind = np.abs(self.z[:,self.solute_loc] - sp2).argmin()
        self.v_sp[t,:] = [self.v[sp1_ind, self.solute_loc], self.v[sp2_ind, self.solute_loc]]
        self.GM_sp[t,:] = [self.GM[t,sp1_ind], self.GM[t,sp2_ind]]
      
      if np.all(~np.isnan([bi1, bi2])):
        bi1_ind = np.abs(self.z[:,self.solute_loc] - bi1).argmin()
        bi2_ind = np.abs(self.z[:,self.solute_loc] - bi2).argmin()
        self.v_bi[t,:] = [self.v[bi1_ind, self.solute_loc], self.v[bi2_ind, self.solute_loc]]
        self.GM_bi[t,:] = [self.GM[t,bi1_ind], self.GM[t,bi2_ind]]

      # calculate I0
      self.I0_arr[t] = calculate_saxs_Io(z=self.z, molar_vol=self.molar_vol, n_electrons=self.n_electrons, Hij=self.det_Hij[t], T=T)
      # get widom line where spinodals are not found, i.e., where mixture is stable
      if np.all(np.isnan(self.x_sp[t])):
        I0_mask = ~np.isnan(self.I0_arr[t])
        if len(I0_mask) > 0:
          # get I0 max
          I0_max = max(self.I0_arr[t][I0_mask])
          # get widom line in mol frac and vol frac
          self.v_I0_max[t] = self.v[I0_mask][np.where(self.I0_arr[t][I0_mask]==I0_max)[0][0]]
          self.x_I0_max[t] = self.z[I0_mask][np.where(self.I0_arr[t][I0_mask]==I0_max)[0][0]]

    # find where there is the smallest difference between the spinodal values
    nan_mask = ~np.isnan(self.x_sp[:,0]) & ~np.isnan(self.x_sp[:,1])
    x_filter = self.x_sp[nan_mask]
    T_values_filter = self.T_values[nan_mask]
    crit_ind = np.abs(x_filter[:,0]-x_filter[:,1]).argmin()
    self.xc = np.mean(x_filter[crit_ind,:])
    self.phic = mol2vol(([self.xc, 1-self.xc]), self.molar_vol)
    self.Tc = T_values_filter[crit_ind]


  def _multicomp_temperature_scaling(self):

    self.GM = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
    self.det_Hij = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
    self.x_bi = []
    self.x_sp = []

    for t, T in enumerate(self.T_values):
      self.model = UNIFAC(T=T, smiles=self.smiles, version=self.unif_version)
      GM = self.model.GM()
      gammas = self.model.gammas()
      det_Hij = self.model.det_Hij()

      self.GM[t] = GM

      bi_vals = binodal_fn(num_comp=self.num_comp, rec_steps=self.model.rec_steps, G_mix=GM, activity_coefs=gammas, solute_idx=self.solute_loc, bi_min=0.001, bi_max=0.99)
      self.x_bi += [bi_vals]

      self.det_Hij[t] = det_Hij
      mask = (det_Hij > -10) & (det_Hij <= 10)
      sps = spinodal_fn(z=self.model.z, Hij=det_Hij)
      self.x_sp += [sps]


  