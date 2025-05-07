import numpy as np
import scipy.optimize as sco 
from scipy.optimize import curve_fit
from scipy import constants
from pathlib import Path
import pandas as pd
from functools import partial

from ..conversions import vol2mol


class FH:
  r"""
  Implements the Flory-Huggins thermodynamic model for binary mixtures.

  :param smiles: SMILES representation of molecules
  :type smiles: str
  :param phi: volume fraction of solute for Smix and Hmix data
  :type phi: numpy.ndarray
  :param Smix: mixing entropy
  :type Smix: numpy.ndarray
  :param Hmix: mixing enthalpy
  :type Hmix: numpy.ndarray
  """
  def __init__(self, smiles: list, phi: np.array, Smix: np.array, Hmix: np.array, V0: np.array):
    # for binary mixtures only!!
    self.R = constants.R / 1000  # kJ/(mol*K)

    self.V0 = V0
    self.V1, self.V2 = self.V0
    self.N0 = [self.V0[i]/min(self.V0) for i in range(len(self.V0))]
    self.N1, self.N2 = self.N0

    self.z = np.load(Path(__file__).parent / "molfr_matrix" / f"{len(smiles)}_molfr_pts.npy")

    self.Smix = Smix
    self.Hmix = Hmix
    self.phi = phi

  @property
  def phic(self):
    r"""
    Volume fraction of solute at critical point

    .. math::
      \phi_c = \frac{1}{\sqrt{N_1} + \sqrt{N_2}}

    .. math::
      N_i = \frac{V_i}{\min(V)}

    where :math:`V_i` is the molar volume of component i, and :math:`V` is a numpy.ndarray containing the molar volumes of all pure components in the mixture.

    :returns: critical volume fraction
    :rtype: float
    """
    return 1/(np.sqrt(self.N1) + np.sqrt(self.N2))

  @property
  def xc(self):
    r"""
    Mol fraction of solute at critical point

    :returns: critical mol fraction
    :rtype: float
    """
    return vol2mol(self.phic, self.V0)

  @property
  def chic(self):
    r"""
    :math:`\chi` parameter at critical point

    .. math::
      \chi_c = \frac{1}{2} \left(\frac{1}{\sqrt{N_1}} + \frac{1}{\sqrt{N_2}} \right)^2

    :returns: critical :math:`\chi` parameter
    :rtype: float
    """
    return 0.5 * (1/np.sqrt(self.N1) + 1/np.sqrt(self.N2))**2

  def scaled_Gmix(self, T):
    r"""
    Estimates the Gibbs mixing free energy as a function of temperature.

    .. math::
      \Delta G_{mix} = \Delta H_{mix} - T \Delta S_{mix}

    :param T: temperature (K)
    :type T: float
    :return: scaled Gibbs mixing free energy
    :rtype: float
    """
    return self.Hmix - T * self.Smix
  
  def chi(self, T):
    r"""
    Calculates the Flory-Huggins interaction parameter :math:`\chi` at a given temperature.
    The interaction parameter is calculated using the Gibbs mixing free energy data.

    :param T: temperature (K)
    :type T: float
    :return: Flory-Huggins interaction parameter
    :rtype: float
    """
    return self.fit_Gmix(phi=self.phi, Gmix=self.scaled_Gmix(T), T=T, V0=self.V0)

  def GM(self, x, T):
    r"""
    Calculates the mixing free energy.

    .. math::
      \Delta G_{mix}^{FH} = RT \left(\frac{\phi \ln{\left(\phi\right)}}{N_1} + \frac{\left(1 - \phi\right) \ln{\left(1 - \phi\right)}}{N_2} \right) + \chi \left(\phi\right) \left(1 - \phi\right)

    :param x: volume fraction of solute
    :type x: float or numpy.ndarray
    :param T: temperature (K)
    :type T: float
    :returns: Gibbs mixing free energy
    :rtype: float or numpy.ndarray
    """
    return self.R * T * (x * np.log(x)/self.N1 + (1-x) * np.log(1-x)/self.N2) + self.chi(T)*x*(1-x)
  
  def dGM_dxs(self, x, T):
    r"""
    Calculates the first derivative of Gibbs mixing free energy with respect to composition.

    .. math::
      \frac{\partial \Delta G_{mix}^{FH}}{\partial \phi_1} = RT \left(\frac{\ln{\left(\phi\right)} + 1}{N_1} - \frac{\ln{\left(1-\phi\right)} + 1}{N_2}\right) + \chi \left(1 - 2\phi \right)

    :param x: volume fraction of solute
    :type x: float or numpy.ndarray
    :param T: temperature (K)
    :type T: float
    :returns: first derivative of Gibbs mixing free energy
    :rtype: float or numpy.ndarray
    """
    return self.R * T * ((np.log(x) + 1)/self.N1 - (np.log(1-x) + 1)/self.N2) + self.chi(T)*(1-2*x)
  
  def det_Hij(self, x, T):
    r"""
    Calculates the determinant of the Hessian of Gibbs mixing free energy, which for a binary system is equivalent to the second derivative of the Gibbs free energy of mixing with respect to composition.

    .. math::
      \frac{\partial^2 \Delta G_{mix}^{FH}}{\partial \phi_1^2} = RT \left(\frac{1}{\phi N_1} + \frac{1}{\left(1 - \phi\right) N_2}\right) - 2 \chi

    :param x: volume fraction of solute
    :type x: float or numpy.ndarray
    :param T: temperature (K)
    :type T: float
    :returns: Hessian of Gibbs mixing free energy
    :rtype: float or numpy.ndarray
    """
    return self.R * T * (1/self.N1/x + 1/self.N2/(1-x)) - 2*self.chi(T)
  
  def Fall(self, x, x0, T):
    r"""
    Calculates the mixing free energy of a two-phase binary mixture.

    .. math::
        x_1, x_2 = x[0], x[1]
    
    .. math::
      d_0 = \frac{x_0 - x_2}{x_1 - x_2}

    .. math::
      \Delta G_{mix}^{FH} = d_0 \cdot \Delta G_{mix}^{FH}(x_1, T) + (1 - d_0) \cdot \Delta G_{mix}^{FH}(x_2, T)
    
    where:

      * :math:`x_1` and :math:`x_2` are the compositions of the two phases
      * :math:`x_0` is the critical composition
      * :math:`d_0` is the partition coefficient
      * :math:`\Delta G_{mix}^{FH}(x, T)` represents the Gibbs free energy of mixing (:func:`GM`) at composition :math:`x` and temperature :math:`T`


    :param x: a list or NumPy array containing the phase compositions [x1, x2]
    :type x: list or numpy.ndarray
    :param x0: estimate for the critical composition (:math:`x_0`)
    :type x0: float
    :param T: temperature (K)
    :type T: float
    :return: Gibbs mixing free energy for the two-phase mixture
    :rtype: float or numpy.ndarray
    """
    x1, x2 = x[0], x[1]
    dc = (x0-x2)/(x1-x2)
    return dc*self.GM(x1, T) + (1-dc)*self.GM(x2, T)

  def Jall(self, x, x0, T):
    r"""
    Calculates the Jacobian vector for a two-phase equilibrium calculation.

    The Jacobian vector (:math:`\text{J}`), representing the derivatives of the phase equilibrium conditions, is calculated as follows:

    .. math::
        x_1, x_2 = x[0], x[1]

    .. math::
        d_0 = \frac{x_0 - x_2}{x_1 - x_2}

    .. math::
        \begin{aligned}
        \text{J} = 
        \begin{bmatrix}
            -\frac{d_0}{x_1 - x_2} \cdot (\Delta G_{mix}^{FH}(x_1, T) - \Delta G_{mix}^{FH}(x_2, T)) + d_0 \cdot \frac{\partial \Delta G_{mix}^{FH}}{\partial \phi}(x_1, T) \\
            \frac{1 - d_0}{x_1 - x_2} \cdot (\Delta G_{mix}^{FH}(x_2, T) - \Delta G_{mix}^{FH}(x_1, T)) + (1 - d_0) \cdot \frac{\partial \Delta G_{mix}^{FH}}{\partial \phi}(x_2, T)
        \end{bmatrix}
        \end{aligned}

    where:

      * :math:`x_1` and :math:`x_2` are the compositions of the two phases
      * :math:`x_0` is the critical composition
      * :math:`d_0` is the partition coefficient
      * :math:`\Delta G_{mix}^{FH}(x, T)` represents the Gibbs free energy of mixing (:func:`GM`) at composition :math:`x` and temperature :math:`T`
      * :math:`\frac{\partial \Delta G_{mix}^{FH}}{\partial \phi}(x, T)` represents the derivative of the Gibbs free energy of mixing with respect to composition (:func:`dGM_dxs`) at :math:`x` and :math:`T`

    :param x: a list or NumPy array containing the phase compositions [x1, x2]
    :type x: list or numpy.ndarray
    :param x0: estimate for the critical composition (:math:`x_0`)
    :type x0: float
    :param T: temperature in Kelvin (K)
    :type T: float
    :return: Jacobian vector, a NumPy array of length 2
    :rtype: numpy.ndarray
    """
    x1, x2 = x[0], x[1]
    dc = (x0-x2)/(x1-x2)
    f1 = self.GM(x1, T)
    f2 = self.GM(x2, T)
    df1 = self.dGM_dxs(x1, T)
    df2 = self.dGM_dxs(x2, T)
    
    return np.array([-dc/(x1-x2)*(f1-f2) + dc*df1, 
            (1-dc)/(x1-x2)*(f2-f1)+(1-dc)*df2])

  def calculate_spinodals_binodals(self, x0, T):
    r"""
    Calculates spinodal and binodal compositions for a binary mixture.

    This function uses numerical methods to determine the spinodal and binodal points
    of a two-phase system.  The spinodal points are found using Brent's method
    to find the roots of the determinant of the Hessian matrix.  The binodal
    points are found by minimizing the system's free energy (:func:`Fall`) to its lowest
    value using the L-BFGS-B algorithm. The Jacobian of Fall (:func:`Jall`) is used
    in the minimization process.

    :param x0: critical composition of the mixture
    :type x0: float
    :param T: temperature (K)
    :type T: float
    :return: A tuple containing the spinodal and binodal compositions: (sp1, sp2, bi1, bi2),
             where sp1 and sp2 are the spinodal compositions, and bi1 and bi2 are
             the binodal compositions.  All compositions are in mol fraction.
    :rtype: tuple(float, float, float, float)
    """
    sp1 = sco.brenth(self.det_Hij, 1e-5, x0, args=(T))  
    sp2 = sco.brenth(self.det_Hij, x0, 0.99, args=(T))

    # Minimize system free energy fall defined above to obtain the binodal volume fractions x
    # via L-BFGS-B algorithm
    bis = sco.minimize(self.Fall, x0=(0.01,0.9), args=(x0, T), 
                        jac=self.Jall, bounds=((1e-5, sp1),(sp2,0.99)), method='L-BFGS-B')

    # all other thermo packages return mol frac--conver vol2mol frac
    sp1 = self.vol2mol(sp1, self.V0)
    sp2 = self.vol2mol(sp2, self.V0)
    bi1 = self.vol2mol(bis.x[0], self.V0)
    bi2 = self.vol2mol(bis.x[1], self.V0)

    return sp1, sp2, bi1, bi2 

  @staticmethod
  def fGM(phi, chi, T, V0):
    r"""
    Calculates the Gibbs free energy of mixing for a binary mixture using the Flory-Huggins model (:func:`GM`).

    :param T: temperature (K)
    :type T: float
    :param phi: volume fraction of solute
    :type phi: float
    :param V0: molar volumes of components
    :type V0: numpy.ndarray
    :param chi: Flory-Huggins interaction parameter
    :type chi: float
    :return: Gibbs mixing free energy
    :rtype: float
    """
    R = constants.R / 1000 # kJ/(mol*K)
    N1, N2 = [V0[i]/min(V0) for i in range(len(V0))]
    return R * T * (phi * np.log(phi)/N1 + (1-phi) * np.log(1-phi)/N2) + chi*phi*(1-phi)

  @staticmethod
  def fit_Gmix(phi, Gmix, T, V0):
    r"""
    Fits the Flory-Huggins model to the Gibbs free energy of mixing data using relationship in :func:`GM`.

    :param T: temperature (K)
    :type T: float
    :param phi: volume fraction of solute
    :type phi: numpy.ndarray
    :param Gmix: Gibbs free energy of mixing
    :type Gmix: numpy.ndarray
    :param V0: molar volumes of components
    :type V0: numpy.ndarray
    :return: fitted :math:`\chi` parameter
    :rtype: float
    """
    fGM_partial = partial(FH.fGM, T=T, V0=V0)
    fit, pcov = curve_fit(fGM_partial, xdata=phi, ydata=Gmix)
    chi = fit[0]
    return chi

  