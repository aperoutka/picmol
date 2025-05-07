import numpy as np
from pathlib import Path
from scipy import constants
from functools import partial
from scipy.optimize import curve_fit

from .cem import PointDisc

class NRTL:
  r"""
  NRTL (Non-Random Two-Liquid) model for binary mixtures.

  This class implements the NRTL activity coefficient model for calculating
  thermodynamic properties of binary mixtures.

  :param IP: a dictionary containing interaction parameters :math:`\tau_{12}` and :math:`\tau_{21}`
  :type IP: dict
  :param z: composition array. If None, it defaults to
      values from PointDisc.
  :type z: numpy.ndarray, optional
  """
  def __init__(self, IP, z=None):
    self.R = constants.R / 1000  # kJ/(mol*K)
    self.tau12, self.tau21 = IP
    self.alpha = 0.2

    self.num_comp = 2
    self.rec_steps = 10

    if z is not None:
      self.z = z
    else:
      point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.rec_steps, load=True, store=False)
      self.z = point_disc.points_mfr

  def G_ij(self, i: int, j: int, T: float):
    r"""
    Calculates the NRTL interaction parameter :math:`G_{ij}`.

    .. math::
        G_{ij} = \exp\left( -\alpha \frac{\tau_{ij}}{RT} \right)

    where:

      * :math:`\alpha` is the non-randomness parameter, set to 0.2
      * :math:`\tau_{ij}` is the interaction parameter between components i and j

    :param i: index of the first component (1 or 2)
    :type i: int
    :param j: index of the second component (1 or 2)
    :type j: int
    :param T: temperature (K)
    :type T: float
    :return: NRTL interaction parameter :math:`G_{ij}`
    :rtype: float
    """
    tau_dict = {1: {2: self.tau12}, 2: {1: self.tau21}}
    return np.exp(-self.alpha * tau_dict[i][j] / (self.R * T))

  def gammas(self, T):
    r"""
    Calculates activity coefficients for the binary mixture.

    .. math::
        \gamma_1 = x_2^2 \left( \tau_{21} \left( \frac{G_{21}}{x_1 + x_2 G_{21}} \right)^2 + \frac{\tau_{12} G_{12}}{(x_2 + x_1 G_{12})^2} \right)

    .. math::
        \gamma_2 = x_1^2 \left( \tau_{12} \left( \frac{G_{12}}{x_2 + x_1 G_{12}} \right)^2 + \frac{\tau_{21} G_{21}}{(x_1 + x_2 G_{21})^2} \right)

    :param T: temperature (K)
    :type T: float
    :return: array of activity coefficients for each component
    :rtype: numpy.ndarray
    """
    G12 = np.exp(-self.alpha*self.tau12/(self.R*T))
    G21 = np.exp(-self.alpha*self.tau21/(self.R*T))
    x1 = self.z[:,0]
    x2 = self.z[:,1]
    gamma_1 = (x2**2) * (self.tau21 * (G21 / (x1 + x2 * G21))**2 + self.tau12*G12/((x2 + x1*G12)**2))
    gamma_2 = (x1**2) * (self.tau12 * (G12 / (x2 + x1 * G12))**2 + self.tau21*G21/((x1 + x2*G21)**2))
    return np.array([gamma_1, gamma_2]).T


  def GM(self, T):
    r"""
    Calculates the Gibbs free energy of mixing for the binary mixture.

    .. math::
        \Delta G_{mix}^{NRTL} = RT \left( x_1 \ln(x_1) + x_2 \ln(x_2) + x_1 x_2 \left( \frac{\tau_{21} G_{21}}{x_1 + x_2 G_{21}} + \frac{\tau_{12} G_{12}}{x_2 + x_1 G_{12}} \right) \right)

    :param T: temperature (K)
    :type T: float
    :return: Gibbs free energy of mixing
    :rtype: numpy.ndarray
    """
    G12 = np.exp(-self.alpha*self.tau12/(self.R*T))
    G21 = np.exp(-self.alpha*self.tau21/(self.R*T))
    x1 = self.z[:,0]
    x2 = self.z[:,1]
    G_ex = -self.R * T * (x1 * x2 * (self.tau21 * G21/(x1 + x2 * G21) + self.tau12 * G12 / (x2 + x1 * G12))) 
    G_id = self.R * T * (x1 * np.log(x1) + x2 * np.log(x2))
    return G_ex + G_id

  def dGM_dxs(self, T):
    r"""
    Calculates the derivative of the Gibbs free energy of mixing with respect to composition.

    .. math::
        \frac{\partial \Delta G_{mix}^{NRTL}}{\partial x_1} = -\left(\frac{n G_{21} - 2 n G_{21} x_1 - n (1 - G_{21}) x_1^2}{(G_{21} + (1 - G_{21}) x_1)^2} + \frac{m - 2 m x_1 - m (G_{12} - 1) x_1^2}{(1 + (G_{12} - 1) x_1)^2}\right) + RT(\ln(x_1) - \ln(x_2))

    .. math::
        n = RT \tau_{21} G_{21}

    .. math::
        m = RT \tau_{12} G_{12}

    :param T: temperature (K)
    :type T: float
    :return: derivative of Gibbs free energy of mixing with respect to composition
    :rtype: numpy.ndarray
    """
    G12 = np.exp(-self.alpha*self.tau12/(self.R*T))
    G21 = np.exp(-self.alpha*self.tau21/(self.R*T))
    x1 = self.z[:,0]
    x2 = self.z[:,1]
    n = self.R * T * self.tau21 * G21
    m = self.R * T * self.tau12 * G12

    return -1*((n*G21 - 2*n*G21*x1 - n*(1-G21)*(x1**2))/((G21 + (1-G21)*x1)**2) + (m - 2*m*x1 - m*(G12-1)*(x1**2))/((1 + (G12-1)*x1)**2)) + self.R*T*(np.log(x1) - np.log(x2))


  def det_Hij(self, T):
    r"""
    Calculates the determinant of the Hessian of Gibbs mixing free energy, which for a binary system is equivalent to the second derivative of the Gibbs free energy of mixing with respect to composition.

    .. math::
        \frac{\partial^2\Delta G_{mix}^{NRTL}}{\partial x_1^2} = 2 \left( \frac{n G_{21}}{(G_{21} + (1 - G_{21}) x_1)^3} + \frac{m G_{12}}{(1 + (G_{12} - 1) x_1)^3} \right) + RT \left( \frac{1}{x_1} + \frac{1}{x_2} \right)

    .. math::
        n = RT \tau_{21} G_{21}

    .. math::
        m = RT \tau_{12} G_{12}

    :param T: temperature (K)
    :type T: float
    :return: second derivative of the Gibbs free energy of mixing
    :rtype: numpy.ndarray
    """
    G12 = np.exp(-self.alpha*self.tau12/(self.R*T))
    G21 = np.exp(-self.alpha*self.tau21/(self.R*T))
    x1 = self.z[:,0]
    x2 = self.z[:,1]
    n = self.R * T * self.tau21 * G21
    m = self.R * T * self.tau12 * G12
    return 2 * ((n*G21/((G21 + (1-G21)*x1)**3)) + (m*G12/((1+(G12-1)*x1)**3))) + self.R*T*(1/x1 + 1/x2)

  @staticmethod
  def fGM(z, tau12, tau21, T):
    r"""
    Calculates the Gibbs free energy of mixing for a binary mixture using the NRTL model (:func:`GM`) for any interaction parameters.

    :param T: temperature (K)
    :type T: float
    :param z: composition array
    :type z: numpy.ndarray
    :param tau12: interaction parameter between components 1 and 2
    :type tau12: float
    :param tau21: interaction parameter between components 2 and 1
    :type tau21: float
    :return: Gibbs free energy of mixing
    :rtype: numpy.ndarray
    """
    alpha = 0.2
    R = constants.R / 1000 # kJ/(mol*K)
    G12 = np.exp(-alpha*tau12/(R*T))
    G21 = np.exp(-alpha*tau21/(R*T))
    x1 = z[:,0]
    x2 = z[:,1]
    G_ex = -R * T * (x1 * x2 * (tau21 * G21/(x1 + x2 * G21) + tau12 * G12 / (x2 + x1 * G12))) 
    G_id = R * T * (x1 * np.log(x1) + x2 * np.log(x2))
    return G_ex + G_id

  @staticmethod
  def fit_Gmix(T, z, Gmix):
    r"""
    Fits the NRTL model to the Gibbs free energy of mixing data using relationship in :func:`GM`.

    :param T: temperature (K)
    :type T: float
    :param Gmix: Gibbs free energy of mixing data
    :type Gmix: numpy.ndarray
    :return: fitted interaction parameters :math:`\tau_{12}` and :math:`\tau_{21}`
    :rtype: dict[str, float]
    """
    fGM_partial = partial(NRTL.fGM, T=T)
    taus, pcov = curve_fit(fGM_partial, z, Gmix)
    return taus