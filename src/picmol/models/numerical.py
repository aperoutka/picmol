import numpy as np
from scipy.optimize import curve_fit
from sympy import symbols, sympify, diff, preorder_traversal, lambdify
from scipy import constants

from ..functions import mol2vol
from .cem import PointDisc


def add_zeros(arr1d, num_comp):
  # add zeros to array
  new_arr = np.zeros(arr1d.size + num_comp)
  new_arr[:len(arr1d)] = arr1d
  return new_arr

def add_pc(z_arr, num_comp):
  # add pure component values to array
  new_arr = np.zeros((z_arr.shape[0] + num_comp, z_arr.shape[1]))
  new_arr[:z_arr.shape[0],:] = z_arr
  pure_comp_arr = []
  for i in range(num_comp):
    a = np.zeros(num_comp)
    a[i] = 1
    pure_comp_arr += [a]
  new_arr[z_arr.shape[0]:,:] = np.array(pure_comp_arr)
  return new_arr


def _get_symbols_sympy_func(expression, xvar: str = 'x'):
  r"""
  Extracts symbols from a SymPy expression, separating 'x' variables.

  This function analyzes a SymPy expression to identify all symbolic variables.
  It then categorizes these symbols into two groups: those starting with 'x'
  (e.g., x1, x2) and all others.  The 'x' symbols are sorted and placed
  first in the returned list.

  :param expression: The SymPy expression to analyze.
  :type expression: sympy.core.expr.Expr
  :param xvar: The prefix used for the 'x' variables.  Defaults to 'x'.
  :type xvar: str, optional

  :returns: A list of strings representing the symbols in the expression.
            The symbols are sorted, with 'x' variables appearing first.
  :rtype: list
    """
  symbols_set = set()
  for arg in preorder_traversal(expression):
    if arg.is_Symbol:
      symbols_set.add(str(arg))
  symbols_list = list(symbols_set)
  x_symbols = sorted([s for s in symbols_list if s.startswith(xvar)])
  other_symbols = sorted([s for s in symbols_list if not s.startswith(xvar)])
  return x_symbols + other_symbols


class QuarticModel:
  r"""
  A class for modeling excess thermodynamic properties using a 4th order Taylor series expansion (:math:`A^E`).

  This class fits a polynomial to enthalpy of mixing (``Hmix_data``) and excess
  entropy (``SE_data``) experimental data. It then uses this fit to calculate various
  thermodynamic properties.

  .. math::
    \begin{align}
      A^{E} = &-\sum_{i=1}^{N-1}\sum_{j=1}^{N} \left[a_{ii} + \cdots + a_{ii\cdots i}\right]x_ix_j \\
      &+ \sum_{i=1}^{N-1}\sum_{j=1}^{N-1} a_{ij}x_ix_j \\
      &+ \sum_{i=1}^{N-1}\sum_{\le j}^{N-1}\sum_{\le k}^{N-1} a_{ijk}x_ix_jx_k \\
      &+ \sum_{i=1}^{N-1}\sum_{\le j}^{N-1}\sum_{\le k}^{N-1}\sum_{\le l}^{N-1} a_{ijkl}x_ix_jx_kx_l 
    \end{align}

  where: 

    * :math:`x_i` are mol fractions of component :math:`i`
    * :math:`a_{ij\ldots}` are fitting parameters

  :param z_data: array of mol fraction data for experimental data
  :type z_data: numpy.ndarray
  :param Hmix_data: enthalpy of mixing data
  :type Hmix_data: numpy.ndarray
  :param SE_data: excess entropy data
  :type SE_data: numpy.ndarray
  :param molar_vol: list of molar volumes
  :type molar_vol: list
  :param z: mol fraction data. Defaults to None.
  :type z: numpy.ndarray, optional
  :param gid_type: type of excess Gibbs energy calculation.
                  Defaults to 'vol'.
  :type gid_type: str, optional
   """
  def __init__(self, z_data, Hmix_data, SE_data, molar_vol, z=None, gid_type='vol'):
    self.molar_vol = molar_vol
    self.Rc = constants.R / 1000
    self.num_comp = len(molar_vol)
    self.rec_steps = 10-self.num_comp
    self.gid_type = gid_type

    self.Hmix_data = add_zeros(Hmix_data, self.num_comp)
    self.SE_data = add_zeros(SE_data, self.num_comp)
    self.z_data = add_pc(z_data, self.num_comp)

    # get mol and vol arrays
    num_pts = {2:10, 3:7, 4:6, 5:5, 6:4}
    self.rec_steps = num_pts[self.num_comp]

    if z is not None:
      self.z = z
    else:
      point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.rec_steps, load=True, store=False)
      self.z = point_disc.points_mfr 

    self.v = mol2vol(self.z, self.molar_vol)


  @property
  def _polynomial_func(self):
    """
    Polynomial function for 4th order Taylor series expansion 
    """
    return self._create_polynomial_for_fitting(self.num_comp)

  @property
  def _df_dx(self):
    """ Property containing the first derivative function"""
    return self._df_dx_func(xvar='x')
  
  @property
  def _d2f_dx2(self):
    """Property containing the second derivative function"""
    return self._d2f_dx2_func(xvar='x')

  @property
  def _dMf_dmi(self):
    """Property containing the derivative on mol fraction basis"""
    return self._df_dx_func(xvar='m')

  def _df_dx_func(self, xvar: str = 'x'):
    """
    Calculates the first derivative of the polynomial function.

    :param xvar: The variable with respect to which the derivative is taken ('x' or 'm').
                  Defaults to 'x'.
    :type xvar: str, optional

    :returns: A dictionary of functions, where each function calculates the
              partial derivative with respect to one of the composition variables.
    :rtype: dict
    """
    # create function from num_components
    py_func = self._create_polynomial_for_sympy(num_comp=self.num_comp, xvar=xvar)
    # convert to sympy function
    sympy_func = self._python_function_to_sympy(python_function=py_func, xvar=xvar)
    # setup variables
    x_symbols = [symbols(f"{xvar}{i}") for i in range(1,self.num_comp+1)]
    if xvar == 'x':
      df_dxs = {i: None for i in range(self.num_comp-1)}
    else:
      df_dxs = {i: None for i in range(self.num_comp)}
    # get list of arguments
    arg_list = _get_symbols_sympy_func(sympy_func, xvar)
    for i in df_dxs.keys():
      df_dx_sympy = diff(sympy_func, x_symbols[i])
      df_dxs[i] = lambdify(arg_list, df_dx_sympy, 'numpy')
    return df_dxs

  def _d2f_dx2_func(self, xvar: str = 'x'):
    """
    Calculates the second derivative of the polynomial function

    :param xvar: The variable with respect to which the derivative is taken ('x' or 'm').
                  Defaults to 'x'.
    :type xvar: str, optional

    :returns: A dictionary of dictionaries of functions, where each function calculates the
              second partial derivative with respect to two of the composition variables
    :rtype: dict
    """
    # create function from num_components
    py_func = self._create_polynomial_for_sympy(num_comp=self.num_comp, xvar=xvar)
    # convert to sympy function
    sympy_func = self._python_function_to_sympy(python_function=py_func, xvar=xvar)
    # setup variables
    x_symbols = [symbols(f"{xvar}{i}") for i in range(1,self.num_comp+1)]
    if xvar == 'x':
      d2f_dx2s = {i: {j: None for j in range(self.num_comp-1)} for i in range(self.num_comp-1)}
    else:
      d2f_dx2s = {i: {j: None for j in range(self.num_comp)} for i in range(self.num_comp)}
    # get list of arguments
    arg_list = _get_symbols_sympy_func(sympy_func, xvar)
    for i in d2f_dx2s.keys():
      df_dx_sympy = diff(sympy_func, x_symbols[i])
      for j in d2f_dx2s[i].keys():
        d2f_dx2_sympy = diff(df_dx_sympy, x_symbols[j])
        d2f_dx2s[i][j] = lambdify(arg_list, d2f_dx2_sympy, 'numpy')
    return d2f_dx2s
  
  def _python_function_to_sympy(self, python_function, xvar: str = 'x'):
    """
    Converts a python function to a sympy expression

    :param python_function: Python function to convert
    :type python_function: Callable
    :param xvar: The variable used in the function. Defaults to 'x'
    :type xvar: str, optional

    :returns: Sympy expression
    :rtype: sympy.Expr
    """
    arg_names = []
    for i in range(1, self.num_comp):
      arg_names.extend([f"a{i}{i}", f"a{i}{i}{i}", f"a{i}{i}{i}{i}"])
    if xvar == 'x':
      x_symbols = [symbols(f"{xvar}{i+1}") for i in range(self.num_comp-1)]
    else:
      x_symbols = [symbols(f"{xvar}{i+1}") for i in range(self.num_comp)]
    arg_symbols = [symbols(name) for name in arg_names]
    # Call the python function with symbolic arguments
    result = python_function(*x_symbols, *arg_symbols)
    # Convert the NumPy array of SymPy symbols to a SymPy expression
    sympy_expression = sympify(result)
    return sympy_expression


  def fit_Hmix(self):
    r"""
    Fits the Taylor series expansion, :math:`A^E`, to ``Hmix_data``.

    This method uses the `curve_fit` function from SciPy to determine the
    optimal parameters for the quartic polynomial that best fits the
    provided ``Hmix_data``.  The fitted parameters are stored in the
    ``Hmix_params`` attribute.

    :return: mixing enthalpy parameters for :math:`A^E`
    :rtype: numpy.ndarray
    """
    fit, pcov = curve_fit(self._polynomial_func, self.z_data, self.Hmix_data)
    self.Hmix_params = fit
    return self.Hmix_params
  
  def fit_SE(self):
    r"""
    Fits the Taylor series expansion, :math:`A^E`, to ``SE_data``.

    This method uses the `curve_fit` function from SciPy to determine the
    optimal parameters for the quartic polynomial that best fits the
    provided ``SE_data``. The fitted parameters are stored in the ``SE_params``
    attribute.

    :return: excess entropy parameters for :math:`A^E`
    :rtype: numpy.ndarray
    """
    fit, pcov = curve_fit(self._polynomial_func, self.z_data, self.SE_data)
    self.SE_params = fit
    return fit

  @property
  def _x_values(self):
    return [self.z[:,i] for i in range(self.num_comp-1)]
  
  @property
  def _m_values(self):
    return [self.z[:,i] for i in range(self.num_comp)]

  def Hmix(self):
    r"""
    Evaluates Taylor series expansion, :math:`A^E`, for mixing enthalpy, :math:`\Delta H_{mix}`, over large composition space.

    :return: enthalpy of mixing
    :rtype: numpy.ndarray
    """
    try:
      self.Hmix_params
    except AttributeError:
      self.fit_Hmix()
    H = self._polynomial_func(self.z, *self.Hmix_params)
    self._Hmix = np.nan_to_num(H, nan=0)
    return self._Hmix

  def SE(self):
    r"""
    Evaluates Taylor series expansion, :math:`A^E`, for excess entropy, :math:`S^E`, over large composition space.

    :return: excess entropy
    :rtype: numpy.ndarray
    """
    try:
      self.SE_params
    except AttributeError:
      self.fit_SE()
    S = self._polynomial_func(self.z, *self.SE_params)
    self._SE = np.nan_to_num(H, nan=0)
    return self._SE

  @property
  def _Gid_symbols(self):
    """symbols for Gid function"""
    return _get_symbols_sympy_func(self._Gid_sympy, xvar='x')

  @property
  def _Gid_sympy(self):
    """get sympy function for Gid"""
    xsum = f"(1"
    for i in range(1, self.num_comp):
      xsum += f"-x{i}"
    xsum += ")"

    vbar = "("
    for i in range(1, self.num_comp):
      vbar += f"x{i}*v{i} + "
    vbar += xsum + f"*v{self.num_comp})"
  
    if self.gid_type == 'vol':
      G = ""
      for i in range(1, self.num_comp):
        G += f"x{i} * ln(x{i}*v{i}/{vbar}) + "
      G += f"{xsum} * ln({xsum}*v{self.num_comp}/{vbar})"
    else:
      G = ""
      for i in range(1, self.num_comp):
        G += f"x{i} * ln(x{i}) + "
      G += f"{xsum} * ln({xsum})"
    return sympify(G)

  def Gid(self, T):
    r"""
    Calculates the ideal Gibbs mixing free energy.

    .. math::
      \frac{G^{id}}{RT} = \sum_i x_i \ln{\left(x_i\right)}

    :param T: temperature (K)
    :type T: float
    :returns: array of calculated Gibbs ideal mixing free energy values
    :rtype: numpy.ndarray
    """
    Gid_py = lambdify(self._Gid_symbols, self._Gid_sympy, 'numpy')
    if self.gid_type == 'vol':
      Gid_calc = self.Rc * T * Gid_py(*self._x_values, *self.molar_vol)
    else:
      Gid_calc = self.Rc * T * Gid_py(*self._x_values)
    return Gid_calc

  @property
  def _Gid_sympy(self):
    """sympy expression for the first derivative of Gid"""
    df_dx = {i: 0 for i in range(self.num_comp-1)}
    for i in range(self.num_comp-1):
      df_dx[i] = diff(self._Gid_sympy, self._Gid_symbols[i])
    return df_dx

  def dGid_dxs(self, T):
    r"""
    Calculates the first derivative of Gibbs ideal mixing free energy with respect to composition.

    .. math::
      \frac{1}{RT}\frac{\partial G^{id}}{\partial x_i} = \ln(x_i) - \ln\left(1 - \sum_{j=1}^{n-1} x_j\right)

    for (i = 1, 2, ..., n-1).

    :param T: temperature (K)
    :type T: float
    :returns: array containing the first derivatives of :func:`Gid` with respect to
              each component's mole fraction.
    :rtype: numpy.ndarray
    """
    df_dx = np.zeros((self.z.shape[0], self.num_comp-1))
    for i in range(self.num_comp-1):
      df_dx_py = lambdify(self._Gid_symbols, self._Gid_sympy[i], 'numpy')
      if self.gid_type == 'vol':
        df_dx[:,i] = self.Rc * T * df_dx_py(*self._x_values, *self.molar_vol)
      else:
        df_dx[:,i] = self.Rc * T * df_dx_py(*self._x_values)
    return df_dx

  @property
  def _d2_Gid_sympy(self):
    """sympy expression for the second derivative of Gid"""
    d2f_dx2 = {i: {j: 0 for j in range(self.num_comp-1)} for i in range(self.num_comp-1)}
    for i in range(self.num_comp-1):
      df_dx_sympy = self._Gid_sympy[i]
      for j in range(self.num_comp-1):
        d2f_dx2[i][j] = diff(df_dx_sympy, self._Gid_symbols[j])
    return d2f_dx2

  def d2Gid_dx2s(self, T):
    r"""
    Calculates the second derivative of Gibbs ideal mixing free energy with respect to composition.

    .. math::
      \frac{1}{RT}\frac{\partial^2 G^{id}}{\partial x_i \partial x_j} =
      \begin{align}
      \begin{cases}
      &\frac{1}{x_i} + \frac{1}{1 - \sum_{k=1}^{n-1} x_k}, & \text{if } i = j \\
      &\frac{1}{1 - \sum_{k=1}^{n-1} x_k}, & \text{if } i \ne j
      \end{cases}
      \end{align}

    :param T: temperature (K)
    :type T: float
    :returns: array containing the second derivatives of :func:`Gid` with respect to
              the mole fractions of each pair of components.
    :rtype: numpy.ndarray
    """
    d2f_dx2 = np.zeros((self.z.shape[0], self.num_comp-1, self.num_comp-1))
    for i in range(self.num_comp-1):
      for j in range(self.num_comp-1):
        d2f_dx2_py = lambdify(self._Gid_symbols, self._d2_Gid_sympy[i][j], 'numpy')
        if self.gid_type == 'vol':
          d2f_dx2[:,i,j] = self.Rc * T * d2f_dx2_py(*self._x_values, *self.molar_vol)
        else:
          d2f_dx2[:,i,j] = self.Rc * T * d2f_dx2_py(*self._x_values)
    return d2f_dx2

  def gammas(self, T):
    r"""
    Calculates the activity coefficients by differentiating excess Gibbs energy, :math:`G^E`, with respect to mol fraction.

    .. math::
      RT \ln \gamma_i = \frac{\partial N G^E}{\partial N_i}

    :param T: temperature (K)
    :type T: float
    :returns: array of the activity coefficients for each
              component as a function of composition.
    :rtype: numpy.ndarray
    """
    try:
      self.Hmix_params
    except AttributeError:
      self.fit_Hmix()
    try:
      self.SE_params
    except AttributeError:
      self.fit_SE()
    ln_gammas = np.zeros((self.z.shape[0], self.num_comp))
    for i in range(self.num_comp):
      ln_gammas[:,i] = self._dMf_dmi[i](*self._m_values, *self.Hmix_params) - T * self._dMf_dmi[i](*self._m_values, *self.SE_params)
    ln_gammas /= self.Rc * T
    return np.exp(ln_gammas)

  def GE(self, T):
    r"""
    Calculates the excess Gibbs energy.

    .. math::
      G^E = \Delta H_{mix} - T S^E

    which is equivalent to:

    .. math::
      G^E = RT \sum_i x_i \ln{\gamma_i}

    :param T: temperature (K)
    :type T: float
    :returns: excess Gibbs energy values.
    :rtype: numpy.ndarray
    """
    try:
      self._Hmix
    except AttributeError:
      self.Hmix()
    try:
      self._SE 
    except AttributeError:
      self.SE()
    ge = self._Hmix - T * self._SE
    ge = np.nan_to_num(ge, nan=0)
    return ge

  def GM(self, T):  
    r"""
    Calculates the Gibbs mixing free energy.

    .. math::
      \begin{align}
        \Delta G_{mix} &= G^E + G^{id} \\
          &= \Delta H_{mix} - T S^E + RT \sum_i x_i \ln{\left(x_i\right)}
      \end{align}

    :param T: temperature (K)
    :type T: float
    :returns: Gibbs mixing free energy values.
    :rtype: numpy.ndarray
    """
    gm = self.GE(T) + self.Gid(T)
    gm = np.nan_to_num(gm, nan=0)
    return gm

  def dGM_dxs(self, T):
    r"""
    Calculates the first derivative of Gibbs mixing free energy by differentiating excess molar properties (:func:`Hmix` and :func:`SE`) defined by :math:`A^E`, with respect to the mol fractions using SymPy.

    .. note::
      This function uses SymPy to perform symbolic differentiation of the excess
      molar property equation. This ensures that the derivatives are exact.
      The partial derivative of :math:`A^E` is taken with respect to each mol fraction :math:`x_i`.

    :param T: temperature (K)
    :type T: float
    :returns: first derivative of :math:`\Delta G_{mix}` with respect to :math:`x` evaluated at temperature (``T``) 
    :rtype: numpy.ndarray
    """
    try:
      self.Hmix_params
    except AttributeError:
      self.fit_Hmix()
    try:
      self.SE_params
    except AttributeError:
      self.fit_SE()
    dgm_arr = np.zeros((self.z.shape[0], self.num_comp-1))
    for i in range(self.num_comp-1):
      dgm = self._df_dx[i](*self._x_values, *self.Hmix_params) - T * self._df_dx[i](*self._x_values, *self.SE_params) + self.dGid_dxs(T)[:,i]
      dgm = np.nan_to_num(dgm, nan=0)
      dgm_arr[:,i] = dgm
    return dgm_arr

  def d2GM_dx2s(self, T):
    r"""
    Calculates the second derivative of Gibbs mixing free energy by differentiating excess molar properties (:func:`Hmix` and :func:`SE`) defined by :math:`A^E`, with respect to the mol fractions using SymPy.

    .. note::
      This function uses SymPy to perform symbolic differentiation of the excess
      molar property equation. This ensures that the derivatives are exact.
      The partial second derivative of :math:`A^E` is taken with respect to mol fractions :math:`x_i` and :math:`x_j`.

    :param T: temperature (K)
    :type T: float
    :returns: second derivative of :math:`\Delta G_{mix}` with respect :math:`x` evaluated at temperature (``T``) 
    :rtype: numpy.ndarray
    """
    try:
      self.Hmix_params
    except AttributeError:
      self.fit_Hmix()
    try:
      self.SE_params
    except AttributeError:
      self.fit_SE()
    d2gm_arr = np.zeros((self.z.shape[0], self.num_comp-1, self.num_comp-1))
    for i in range(self.num_comp-1):
      for j in range(self.num_comp-1):
        d2gm = self._d2f_dx2[i][j](*self._x_values, *self.Hmix_params) - T * self._d2f_dx2[i][j](*self._x_values, *self.SE_params) + self.d2Gid_dx2s(T)[:,i,j]
        d2gm = np.nan_to_num(d2gm, nan=0)
        d2gm_arr[:,i,j] = d2gm
    return d2gm_arr

  def det_Hij(self, T):
    r"""
    Calculates the determinant of the Hessian matrix of :math:`\Delta G_{mix}`.

    :param T: temperature (K)
    :type T: float

    :returns: The determinant of the Hessian matrix of the second derivatives
              of the excess Gibbs free energy with respect to composition.
    :rtype: numpy.ndarray
    """
    return np.linalg.det(self.d2GM_dx2s(T))


  @staticmethod
  def _create_polynomial_for_fitting(num_comp):
    r"""Creates the polynomial for fitting"""
    # generate argument names
    arg_names = []
    for i in range(1, num_comp):
      arg_names.extend([f"a{i}{i}",f"a{i}{i}{i}",f"a{i}{i}{i}{i}"])
    # include 'z' at beginning of arg_list
    arg_list = ', '.join(['z'] + arg_names)

    # build the function code as string
    func_code = f"""
def quartic_taylor_expansion({arg_list}):
  N = z.shape[1]
  args = [{', '.join(arg_names)}]

  f = np.zeros(z.shape[0])

  for i in range(1, N):
    ai = args[i - 1] + args[i] + args[i + 1]
    for j in range(1, N+1):
      f -= (ai) * z[:,i-1] * z[:,j-1]

  for i in range(1, N):
    for j in range(1, N):
      if i <= j:
        f += args[i - 1] * z[:,i-1] * z[:,j-1]
        for k in range(1, N):
          if j <= k:
            f += args[i] * z[:,i-1] * z[:,j-1] * z[:,k-1]
            for l in range(1, N):
              if k <= l:
                f += args[i + 1] * z[:,i-1] * z[:,j-1] * z[:,k-1] * z[:,l-1]

  return f
    """
    # Create a local namespace for the function definition
    func_namespace = {}
    # Include '__name__' in the globals dictionary
    exec(func_code, globals(), func_namespace)
    # Retrieve the dynamically created function
    quartic_function = func_namespace['quartic_taylor_expansion']
    return quartic_function

  @staticmethod
  def _create_polynomial_for_sympy(num_comp: int, xvar: str = 'x'):
    r"""create polynomial for sympy analysis"""
    # generate argument names
    arg_names = []
    for i in range(1, num_comp):
        arg_names.extend([f"a{i}{i}", f"a{i}{i}{i}", f"a{i}{i}{i}{i}"])
    # create x variables string
    if xvar == 'x':
      num_vars = num_comp-1
      x_vars_str = ', '.join([f'{xvar}{i+1}' for i in range(num_comp-1)])
    else:
      num_vars = num_comp
      x_vars_str = ', '.join([f'{xvar}{i+1}' for i in range(num_comp)])

    # build the function code as string
    func_code = f"""
def quartic_taylor_expansion({x_vars_str}, {', '.join(arg_names)}):
  args = [{', '.join(arg_names)}]

  if {num_vars} < int({num_comp}):
    M = 1
    x = np.array([{', '.join([f'x{i+1}' for i in range(num_comp-1)])}, 1 - sum([{', '.join([f'x{i+1}' for i in range(num_comp-1)])}])])
  else:
    M = sum([{', '.join([f'm{i+1}' for i in range(num_comp)])}])
    x = np.array([{', '.join([f'm{i+1}' for i in range(num_comp)])}])
  
  N = len(x)
  f = 0

  for i in range(1, N):
    ai = args[i - 1] + args[i] + args[i + 1]
    for j in range(1, N+1):
      f -= (ai) * x[i-1] * x[j-1] / M
  
  for i in range(1, N):
    for j in range(1, N):
      if i <= j:
        f += args[i - 1] * x[i-1] * x[j-1] / M
        for k in range(1, N):
          if j <= k:
            f += args[i] * x[i-1] * x[j-1] * x[k-1] / (M**2)
            for l in range(1, N):
              if k <= l:
                f += args[i + 1] * x[i-1] * x[j-1] * x[k-1] * x[l-1] / (M**3)
  
  return f
    """
    # Create a local namespace for the function definition
    func_namespace = {}
    # Include '__name__' in the globals dictionary
    exec(func_code, globals(), func_namespace)
    # Retrieve the dynamically created function
    quartic_function = func_namespace['quartic_taylor_expansion']
    return quartic_function