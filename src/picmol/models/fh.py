import numpy as np
import scipy.optimize as sco 
from scipy.optimize import curve_fit
from pathlib import Path
import pandas as pd
from functools import partial

from ..get_molecular_properties import load_molecular_properties

# def get_molelecular_properties(smiles):
#   file = Path(__file__).parent.parent / "data" / "molecular_properties.csv"
#   df = pd.read_csv(file).set_index("smiles")
#   return df.loc[smiles, :]

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
  def __init__(self, smiles: list, phi: np.array, Smix: np.array, Hmix: np.array):
    """ for binary mixtures only!! """
    self.R = 8.314E-3

    df_mol_prop = load_molecular_properties(smiles, 'smiles')

    self.V0 = df_mol_prop["molar_vol"].to_numpy()
    self.V1, self.V2 = self.V0
    self.N0 = [self.V0[i]/min(self.V0) for i in range(len(self.V0))]
    self.N1, self.N2 = self.N0

    self.z = np.load(Path(__file__).parent / "molfr_matrix" / f"{len(smiles)}_molfr_pts.npy")

    self.Smix = Smix
    self.Hmix = Hmix
    self.phi = phi

  def fit_chi(self, T):
    r"""
    Fits the Flory-Huggins interaction parameter (:math:`\chi`) to experimental Gibbs mixing free energy at a given temperature.

    :param T: temperature (K)
    :type T: float
    :return: fitted :math:`\chi` parameter
    :rtype: float
    """
    G = self.Hmix - T * self.Smix

    def GM_fit(x, chi, Tx):
      # for fitting chi
      return 8.314E-3 * Tx * (x * np.log(x)/self.N1 + (1-x) * np.log(1-x)/self.N2) + chi*x*(1-x)

    GM_fixed_T = partial(GM_fit, Tx=T)

    fit, pcov = curve_fit(GM_fixed_T, xdata=self.phi, ydata=G)
    chi = fit[0]
    return chi

  def vol2mol(self, phi):
    r"""
    Converts volume fraction to mole fraction, where :math:`V_i` is molar volume of component i.

    .. math::
      x = \frac{\phi \left(\frac{V_2}{V_1}\right)}{1 - \phi + \phi \left(\frac{V_2}{V_1}\right)}
    
    :param phi: volume fraction
    :type phi: float
    :return: mol fraction
    :rtype: float
    """
    return phi * (self.V2/self.V1)/(1 - phi + phi*(self.V2/self.V1))

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
    return vol2mol(self.phic)

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
    chi = self.fit_chi(T)
    return self.R * T * (x * np.log(x)/self.N1 + (1-x) * np.log(1-x)/self.N2) + chi*x*(1-x)
  
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
    chi = self.fit_chi(T)
    return self.R * T * ((np.log(x) + 1)/self.N1 - (np.log(1-x) + 1)/self.N2) + chi*(1-2*x)
  
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
    chi = self.fit_chi(T)
    return self.R * T * (1/self.N1/x + 1/self.N2/(1-x)) - 2*chi
  
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
    sp1 = self.vol2mol(sp1)
    sp2 = self.vol2mol(sp2)
    bi1 = self.vol2mol(bis.x[0])
    bi2 = self.vol2mol(bis.x[1])

    return sp1, sp2, bi1, bi2 
