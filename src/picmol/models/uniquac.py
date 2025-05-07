import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import constants

from .unifac_subgroups.fragmentation import Groups
from .cem import PointDisc

def UNIQUAC_R(smiles: list):
  r"""
  Calculates the UNIQUAC r parameter for a list of molecules.

  The r parameter represents the relative size of a molecule in the UNIQUAC model.
  It is calculated by summing the r parameters of the subgroups (*k*) present in the molecule.

  The UNIQUAC r parameter for molecule *i* is calculated using the following equation:

  .. math::
      r_i = \sum_k \nu_{ki} r_k

  where:

    * :math:`\nu_{ki}` is the number of occurrences of subgroup k in molecule i

  :param smiles: a list of SMILES strings representing the molecules
  :type smiles: list
  :return: array containing the r parameters for each molecule
  :rtype: numpy.ndarray
  """
  g = [Groups(smile) for smile in smiles]
  r = np.array([gg.unifac.r for gg in g])
  return r

def UNIQUAC_Q(smiles: list):
  r"""
  Calculates the UNIQUAC q parameter for a list of molecules.

  The q parameter represents the relative surface area of a molecule in the UNIQUAC model.
  It is calculated by summing the q parameters of the subgroups (*k*) present in the molecule.

  The UNIQUAC q parameter is calculated for molecule *i* using the following equation:

  .. math::
      q_i = \sum_k \nu_{ki} q_k
  
  where:

    * :math:`\nu_{ki}` is the number of occurrences of subgroup k in molecule i

  :param smiles: list of SMILES strings representing the molecules
  :type smiles: list
  :return: array containing the q parameters for each molecule
  :rtype: numpy.ndarray
  """
  g = [Groups(smile) for smile in smiles]
  q = np.array([gg.unifac.q for gg in g])
  return q





class UNIQUAC:
  r"""
  UNIQUAC (UNIversal QUAsiChemical) model class.

  This class implements the UNIQUAC activity coefficient model for calculating
  thermodynamic properties of multi-component mixtures.

  :param IP: array of interaction parameters (:math:`\Delta u_{ij}`).
  :type IP: numpy.ndarray
  :param smiles: list of SMILES strings representing the molecules in the mixture.
  :type smiles: list
  :param z: composition array (mol fractions). If None, it defaults to
      values from PointDisc.
  :type z: numpy.ndarray, optional
  :param r: UNIQUAC r parameters for each component. If None, it will be
      calculated from smiles.
  :type r: numpy.ndarray, optional
  :param q: UNIQUAC q parameters for each component. If None, it will be
    calculated from smiles.
  :type q: numpy.ndarray, optional
  """
    
  def __init__(self, IP, smiles, z=None, r=None, q=None):
    self.du = IP
    self.Rc = constants.R / 1000
    self.N = len(smiles)

    num_pts = {
      2:10, 3:7, 4:6, 5:5, 6:4
    }
    self.num_comp = len(smiles)
    self.rec_steps = num_pts[self.num_comp]

    if z is not None:
      self.z = z
    else:
      point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.rec_steps, load=True, store=False)
      self.z = point_disc.points_mfr    
    
    self.smiles = smiles

    if r is not None and q is not None:
      self._r = r
      self._q = q

  @property 
  def zc(self):
    r"""    
    :return: coordination number, set to 10
    :rtype: float
    """
    return 10.

  @property
  def r(self):
    r"""
    Retrieves the r parameter (i.e., relative size of a molecule) for each component from :func:`UNIQUAC_R`.

    :return: array of r parameters for each component
    :rtype: numpy.ndarray
    """
    try:
      self._r 
    except AttributeError:
      self._r = UNIQUAC_R(self.smiles)
    return self._r

  @property
  def q(self):
    r"""
    Retrieves the q parameter (i.e., the relative surface area of a molecule) for each component from :func:`UNIQUAC_Q`.

    :return: array of q parameters for each component
    :rtype: numpy.ndarray
    """
    try:
      self._q
    except AttributeError:
      self._q = UNIQUAC_Q(self.smiles)
    return self._q

  @property
  def l(self):
    r"""
    Calculates the UNIQUAC parameter l for component :math:`i` from ``r`` and ``q`` parameters using the coordination number z\ :sub:`c`.

    .. math::
        l_i =  \frac{z_c}{2}(r_i - q_i) - (r_i - 1)    

    :return: array of l parameters for each component
    :rtype: numpy.ndarray
    """
    return (self.zc / 2) * (self.r - self.q) - (self.r - 1)

  @property
  def rbar(self):
    r"""
    Calculates the mixture average ``r`` parameter.

    .. math::
        \overline{r} = \sum_i^n x_i r_i

    :return: linear combination of ``r`` parameter for each component in mixture
    :rtype: numpy.ndarray
    """
    return self.z @ self.r
  
  @property
  def qbar(self):
    r"""
    Calculates the mixture average ``q`` parameter.
    
    .. math::
        \overline{q} = \sum_i^n x_i q_i

    :return: linear combination of ``q`` parameter for each component in mixture
    :rtype: numpy.ndarray
    """
    return self.z @ self.q
  
  @property
  def lbar(self):
    r"""
    Calculates the mixture average ``l`` parameter.

    .. math::
        \overline{l} = \sum_i^n x_i l_i

    :return: linear combination of ``l`` parameter for each component in mixture
    :rtype: numpy.ndarray
    """
    return self.z @ self.l

  @property
  def rr(self):
    return np.ones((self.z.shape[0], 1)) * self.r

  @property
  def qq(self):
    return np.ones((self.z.shape[0], 1)) * self.q
  
  @property
  def ll(self):
    return np.ones((self.z.shape[0], 1)) * self.l

  @property
  def phi(self):
    r"""
    Calculates the segment fraction :math:`\phi` for component :math:`i`.

    .. math::
        \phi_i = \frac{x_i r_i}{\bar{r}}

    :return: array of segment fractions
    :rtype: numpy.ndarray
    """
    return self.z * (1 / self.rbar[:, None]) * self.r

  @property
  def theta(self):
    r"""
    Calculates the area fraction :math:`\theta` for component :math:`i`.
    
    .. math::
        \theta_i = \frac{x_i q_i}{\bar{q}}

    :return: array of area fractions
    :rtype: numpy.ndarray
    """
    return self.z * (1 / self.qbar[:, None]) * self.q
  
  def tau(self, T):
    r"""
    Calculates the UNIQUAC interaction parameter :math:`\tau` between components :math:`i` and :math:`j`.

    .. math::
        \tau_{ij} = \exp\left( -\frac{\Delta u_{ij}}{RT} \right)

    :param T: temperature (K)
    :type T: float
    :return: array of :math:`\tau` parameters
    :rtype: numpy.ndarray
    """
    return np.exp(-self.du / (self.Rc * T))

  def rho(self, T):
    r"""
    Calculates the :math:`\rho` parameter for component :math:`i`.

    .. math::
        \rho_i = \sum_j^n \theta_j \tau_{ji}

    :param T: temperature (K)
    :type T: float
    :return: array of :math:`\rho` parameters
    :rtype: numpy.ndarray
    """
    return self.theta @ self.tau(T)

  def Gid(self, T):
    r"""
    Calculates the ideal Gibbs energy of mixing.
    
    .. math::
        \frac{G^{id}}{RT} = \sum_i^n x_i \ln x_i

    :param T: temperature (K)
    :type T: float
    :return: ideal Gibbs energy of mixing
    :rtype: numpy.ndarray
    """
    return self.Rc * T * np.sum(self.z * np.log(self.z), axis=1)

  def GE(self, T):
    r"""
    Calculates the excess Gibbs energy.

    .. math::
         \frac{G^E}{RT} = \sum_i^n x_i \ln \frac{\phi_i}{x_i} + \frac{z_c}{2} \sum_i^n x_i q_i \ln \frac{\theta_i}{\phi_i} - \sum_i^n \left( x_i q_i \ln \sum_j^n \theta_j \tau_{ji} \right)

    :param T: temperature (K)
    :type T: float
    :return: total excess Gibbs energy
    :rtype: numpy.ndarray
    """
    ge_res = -self.Rc * T * (self.z * np.log(np.dot(self.theta , self.tau(T)))) @ self.q
    ge_comb = self.Rc * T * (np.sum(self.z * np.log(self.phi/self.z), axis=1) + (self.zc / 2) * (self.z * (np.log(self.theta/self.phi))) @ self.q)
    return np.nan_to_num(ge_res + ge_comb)
  
  def GM(self, T):
    r"""
    Calculates the total Gibbs free energy of mixing.

    .. math::
      \begin{align}
        \frac{\Delta G_{mix}}{RT} &= \frac{G^{id}}{RT} + \frac{G^E}{RT} \\
        &= \sum_i^n x_i \ln \phi_i + \frac{z_c}{2} \sum_i^n x_i q_i \ln \frac{\theta_i}{\phi_i} - \sum_i^n \left( x_i q_i \ln \sum_j^n \theta_j \tau_{ji} \right)
      \end{align}

    :param T: temperature (K)
    :type T: float
    :return: total Gibbs free energy of mixing
    :rtype: numpy.ndarray
    """
    return np.nan_to_num(self.GE(T) + self.Gid(T))
  
  def gammas(self, T):
    r"""
    Calculates the activity coefficients of each component in the mixture.

    .. math::
        \ln \gamma_i = \ln \frac{\phi_i}{x_i} + \frac{z_c}{2} q_i \ln \frac{\theta_i}{\phi_i} + l_i - \frac{\phi_i}{x_i} \bar{l} - q_i \ln \sum_j^n \theta_j \tau_{ji} + q_i - q_i \sum_j^n \frac{\theta_j \tau_{ji}}{\sum_k^n \theta_k \tau_{kj}}

    :param T: temperature (K)
    :type T: float or numpy.ndarray
    :return: activity coefficients of each component
    :rtype: numpy.ndarray
    """

    theta_tau_j = np.zeros(self.z.shape)
    for i in range(self.z.shape[1]):
      theta_tau = np.zeros(self.z.shape[0])
      for j in range(self.z.shape[1]):
        theta_tau += self.theta[:,j] * self.tau(T)[i,j] / self.rho(T)[:,j]
      theta_tau_j[:,i] = theta_tau

    gammas = np.log(self.phi/self.z) + (self.zc / 2) * self.qq * (np.log(self.theta/self.phi)) + self.ll - (self.phi/self.z) * (self.lbar[:, None]) -  self.qq * np.log(np.dot(self.theta , self.tau(T))) + self.qq - self.qq *  theta_tau_j

    return gammas
  

  def mu(self, T):
    r"""
    Calculates the chemical potential of each component in the mixture.

    .. math::
        \frac{\mu_i}{RT} = \ln \phi_i + \frac{z_c}{2} q_i \ln{\frac{\theta_i}{\phi_i}} + l_i - \overline{l}\frac{\phi_i}{x_i} - q_i \ln \sum_{j=1}^n \theta_j \tau_{ji} + q_i - q_i \sum_{j=1}^n \left(\frac{\theta_j \tau_{ij}}{\sum_{k=1}^n \theta_k \tau_{kj}}\right)

    :param T: temperature (K)
    :type T: float
    :return: chemical potential of each component
    :rtype: numpy.ndarray
    """

    mu_comb = np.log(self.phi) + (self.zc / 2) * self.qq * np.log(self.theta/self.phi) + self.ll - (self.lbar[:, None] @ np.array([np.ones(np.shape(self.z)[1])])) * self.phi / self.z

    mu_res = - self.qq * (np.log(np.dot(self.theta , self.tau(T)))) + self.qq - self.qq * (self.theta / (np.dot(self.theta , self.tau(T)))) @ self.tau(T)

    return self.Rc * T * (mu_comb + mu_res)
  

  def dGM_dxs(self, T):
    r"""
    Calculates the first derivative of the Gibbs free energy of mixing
    with respect to the mol fractions.

    .. math::
        \frac{\partial \Delta G_{mix}}{\partial x_i} = \mu_i - \mu_n

    :param T: temperature (K)
    :type T: float or numpy.ndarray
    :return: first derivative of mixing free energy
    :rtype: numpy.ndarray
    """
    mu = self.mu(T)
    dGM_dxs = np.zeros((self.z.shape[0], self.z.shape[1]-1))
    for i in range(self.z.shape[1]-1):
      dGM_dxs[:,i] = mu[:,i] - mu[:,self.z.shape[1]-1]
    return dGM_dxs


  def dmu_dxs(self, T):
    r"""
    Calculates the derivative of the chemical potential with respect to the mol fractions.

    .. math::
        \frac{1}{RT}\frac{\partial \mu_i}{\partial x_j} = 
        \frac{1}{\phi_i} \frac{\partial \phi_i}{\partial x_j} \left( 1 - q_i \frac{z_c}{2} \right) + \frac{1}{\theta_i} \frac{\partial \theta_i}{\partial x_j} q_i \frac{z_c}{2}
        - \overline{l} \left( \frac{1}{x_i} \frac{\partial \phi_i}{\partial x_j} - \frac{\delta_{ij} \phi_i}{x_j^2} \right) 
        + \frac{q_i q_j}{\overline{q}} \left( 1 - \frac{\tau_{ji}}{\sum_k^n \theta_k \tau_{kj}} - \frac{\tau_{ij}}{\sum_k^n \theta_k \tau_{ki}} + \sum_l^n \frac{\theta_l \tau_{jl} \tau_{il}}{\left( \sum_k^n \theta_k \tau_{kl} \right)^2} \right)

    :param T: temperature (K)
    :type T: float or numpy.ndarray
    :return: second derivative of Gibbs free energy / derivative of
              chemical potential
    :rtype: numpy.ndarray
    """
    ones_2darr = np.array([np.ones(np.shape(self.z)[1])])
    dmu_dxs_comb = np.zeros((np.shape(self.z)[0], np.shape(self.z)[1], np.shape(self.z)[1]))

    for ii in range(np.shape(self.z)[1]):
      i_arr = np.array([np.zeros(np.shape(self.z)[1])])
      i_arr[0][ii] = 1

      dphi_dz1 = (self.rbar[:, None] * (self.r[None, :] * i_arr) - self.rr * self.z * self.r[ii]) / (self.rbar[:, None] ** 2 @ ones_2darr)
      dtheta_dz1 = (self.qbar[:, None] * (self.q[None, :] * i_arr) - self.qq * self.z * self.q[ii]) / (self.qbar[:, None] ** 2 @ ones_2darr)

      dmu_comb1_dz1 = ((1 -(self.zc / 2) * self.qq) * dphi_dz1 / self.phi + (self.zc / 2) * self.qq * dtheta_dz1 / self.theta)

      dmu_comb2_dz1 = -((self.lbar[:, None] @ ones_2darr) * (dphi_dz1 / self.z - self.phi * (np.ones((self.z.shape[0], 1)) @ i_arr) / self.z ** 2) + (self.phi / self.z) * self.l[ii])

      dmu_dxsi = dmu_comb1_dz1 + dmu_comb2_dz1
      dmu_dxs_comb[:,ii,:] = dmu_dxsi
    
    dmu_dxs_res = np.zeros((np.shape(self.z)[0], np.shape(self.z)[1], np.shape(self.z)[1]))
    for ii in range(np.shape(self.z)[1]):
      for jj in range(np.shape(self.z)[1]):
        dmu_dxs_res[:,ii,jj] = ((self.q[ii]*self.q[jj])/(self.qbar)) * (1 - (self.tau(T)[jj][ii]/self.rho(T)[:,ii]) - (self.tau(T)[ii][jj]/self.rho(T)[:,jj]) + (((self.theta * self.tau(T)[ii] )/ self.rho(T)**2) @ self.tau(T)[jj]))
    
    dmu_dxs = self.Rc * T * (dmu_dxs_comb + dmu_dxs_res)
    return dmu_dxs


  def Hij(self, T):
    r"""
    Calculates the Hessian matrix, with elements :math:`H_{ij}`, of the Gibbs free energy of mixing
    with respect to the mol fractions.

    .. math::
        H_{ij} = \frac{\partial \mu_i}{\partial x_j} - \frac{\partial \mu_n}{\partial x_j} - \frac{\partial \mu_i}{\partial x_n} + \frac{\partial \mu_n}{\partial x_n}

    :param T: temperature (K)
    :type T: float or numpy.ndarray
    :return: Hessian matrix of the mixing free energy
    :rtype: numpy.ndarray
    """
    dmu_dxs = self.dmu_dxs(T)
    Hij = np.empty((np.shape(self.z)[0], np.shape(self.z)[1]-1, np.shape(self.z)[1]-1))
    n = np.shape(self.z)[1]-1
    for ii in range(np.shape(self.z)[1]-1):
      for jj in range(np.shape(self.z)[1]-1):
        Hij[:,ii,jj] = dmu_dxs[:,ii,jj]-dmu_dxs[:,ii,n]-dmu_dxs[:,n,jj]+dmu_dxs[:,n,n]
    return Hij
  

  def det_Hij(self, T):
    r"""
    Calculates the determinant of the Hessian matrix of the Gibbs free energy
    of mixing.

    :param T: temperature (K)
    :type T: float or numpy.ndarray
    :return: Determinant of the Hessian matrix.
              Shape is (number of conditions,).
    :rtype: numpy.ndarray
    """
    return np.linalg.det(self.Hij(T))

  @staticmethod
  def fHmix(T, z, du, smiles):
    r"""
    Calculates the residual contributon to Gibbs mixing free energy, i.e., enthalpy of mixing using the UNIQUAC model.

    .. math::
        \frac{\Delta H_{mix}}{RT} = -\sum_i^n \left( x_i q_i \ln \sum_j^n \theta_j \tau_{ji} \right)
 
    :param T: temperature (K)
    :type T: float
    :param z: composition array
    :type z: numpy.ndarray
    :param du: interaction parameters (:math:`\Delta u_{ij}`)
    :type du: numpy.ndarray
    :param smiles: list of SMILES strings representing the molecules
    :type smiles: list
    :return: Gibbs free energy of mixing
    :rtype: numpy.ndarray
    """
    N_mols = z.shape[1]
    r = UNIQUAC_R(smiles)
    q = UNIQUAC_Q(smiles)

    Rc = 8.314E-3
    zc = 10
    l = (zc / 2) * (r - q) - (r - 1)
    tau = np.exp(-du / (Rc * T))
    rbar = z @ r
    qbar = z @ q
    # Calculate phi and theta (segment and area fractions)
    phi = z * (1 / rbar[:, None]) * r
    theta = z * (1 / qbar[:, None]) * q
    # Solve for free energy of mixing GM 
    GM_res = -Rc * T * (z * np.log(np.dot(theta , tau))) @ q
    return GM_res

  @staticmethod
  def fSmix(T, z, du, smiles):
    r"""
    Calculates the combinatorial contribution to Gibbs mixing free energy, i.e., entropy of mixing using the UNIQUAC model.

    .. math::
        \frac{\Delta S_{mix}}{RT} = \sum_i^n x_i \ln \phi_i + \frac{z_c}{2} \sum_i^n x_i q_i \ln \frac{\theta_i}{\phi_i} 

    :param T: temperature (K)
    :type T: float
    :param z: composition array
    :type z: numpy.ndarray
    :param du: interaction parameters (:math:`\Delta u_{ij}`)
    :type du: numpy.ndarray
    :param smiles: list of SMILES strings representing the molecules
    :type smiles: list
    :return: Gibbs free energy of mixing
    :rtype: numpy.ndarray
    """
    N_mols = z.shape[1]
    r = UNIQUAC_R(smiles)
    q = UNIQUAC_Q(smiles)
    Rc = 8.314E-3
    zc = 10
    rbar = z @ r
    qbar = z @ q
    # Calculate phi and theta (segment and area fractions)
    phi = z * (1 / rbar[:, None]) * r
    theta = z * (1 / qbar[:, None]) * q
    # Solve for free energy of mixing GM 
    GM_comb = Rc * T * (np.sum(z * np.log(phi), axis=1) + (zc / 2) * (z * (np.log(theta/phi))) @ q)
    return GM_comb

 
  @staticmethod
  def fGM(T, z, du, smiles):
    r"""
    Calculates the Gibbs free energy of mixing (:func:`GM`) using the UNIQUAC model.

    :param T: temperature (K)
    :type T: float
    :param z: composition array
    :type z: numpy.ndarray
    :param du: interaction parameters (:math:`\Delta u_{ij}`)
    :type du: numpy.ndarray
    :return: Gibbs free energy of mixing
    :rtype: numpy.ndarray
    """
    N_mols = z.shape[1]
    r = UNIQUAC_R(smiles)
    q = UNIQUAC_Q(smiles)
    Rc = 8.314E-3
    zc = 10
    l = (zc / 2) * (r - q) - (r - 1)
    tau = np.exp(-du / (Rc * T))
    rbar = z @ r
    qbar = z @ q
    # Calculate phi and theta (segment and area fractions)
    phi = z * (1 / rbar[:, None]) * r
    theta = z * (1 / qbar[:, None]) * q
    # Solve for free energy of mixing GM 
    GM_comb = Rc * T * (np.sum(z * np.log(phi), axis=1) + (zc / 2) * (z * (np.log(theta) - np.log(phi))) @ q)
    GM_res = -Rc * T * (z * np.log(np.dot(theta , tau))) @ q
    return GM_comb + GM_res

  @staticmethod
  def _create_du_matrix(N_mols, du_flat):
    r"""
    Creates a square matrix of interaction parameters (:math:`\Delta u_{ij}`) from a flat array.

    :param N_mols: number of components in the mixture
    :type N_mols: int
    :param du_flat: flat array of interaction parameters
    :type du_flat: numpy.ndarray
    :return: square matrix of interaction parameters
    :rtype: numpy.ndarray
    """
    du = np.zeros((N_mols, N_mols))
    ij = 0
    for i in range(N_mols):
      for j in range(N_mols):
        if i < j:
          du[i,j] = du_flat[ij]
          du[j,i] = du_flat[ij]
          ij += 1
    return du

  @staticmethod
  def fit_Hmix(T, Hmix, z, smiles):
    r"""
    Fits UNIQUAC interaction parameters (:math:`\Delta u_{ij}`) to experimental enthalpy of mixing (``Hmix``) data using :func:`fHmix`.

    :param T: temperature (K)
    :type T: float
    :param Hmix: array of experimental enthalpy of mixing values
    :type Hmix: numpy.ndarray
    :param z: array of mol fractions
    :type z: numpy.ndarray
    :param smiles: list of SMILES strings representing the molecules
    :type smiles: list
    :return: array of fitted UNIQUAC interaction parameters (:math:`\Delta u_{ij}`)
    :rtype: numpy.ndarray
    """
    N_mols = z.shape[1]
    def hmix_to_fit(z, *du_flat):
      """wrapper function for curve_fit to calculate Hmix."""
      du = UNIQUAC._create_du_matrix(N_mols, du_flat)
      return UNIQUAC.fHmix(T, z, du, smiles)
    # fit IP to Hmix
    ij_combo = N_mols * (N_mols - 1) // 2 # number of unique pairs
    popt, pcov = curve_fit(hmix_to_fit, z, Hmix, p0=np.zeros(ij_combo), bounds=(-np.inf, np.inf))
    # Reshape the fitted parameters into a square matrix
    du_fitted = UNIQUAC._create_du_matrix(N_mols, popt)
    return du_fitted

  @staticmethod
  def fit_Gmix(T, Gmix, z, smiles):
    r"""
    Fits UNIQUAC interaction parameters (:math:`\Delta u_{ij}`) to experimental Gibbs mixing (``Gmix``) data using :func:`fGM`.

    :param T: temperature (K)
    :type T: float
    :param Gmix: array of experimental Gibbs mixing free energy values
    :type Gmix: numpy.ndarray
    :param z: array of mol fractions
    :type z: numpy.ndarray
    :param smiles: list of SMILES strings representing the molecules
    :type smiles: list
    :return: array of fitted UNIQUAC interaction parameters (:math:`\Delta u_{ij}`)
    :rtype: numpy.ndarray
    """
    N_mols = z.shape[1]
    def gmix_to_fit(z, *du_flat):
      """wrapper function for curve_fit to calculate Hmix."""
      du = UNIQUAC._create_du_matrix(N_mols, du_flat)
      return UNIQUAC.fGM(T, z, du, smiles)
    # fit IP to Hmix
    ij_combo = N_mols * (N_mols - 1) // 2 # number of unique pairs
    popt, pcov = curve_fit(gmix_to_fit, z, Gmix, p0=np.zeros(ij_combo), bounds=(-np.inf, np.inf))
    # Reshape the fitted parameters into a square matrix
    du_fitted = UNIQUAC._create_du_matrix(N_mols, popt)
    return du_fitted    
