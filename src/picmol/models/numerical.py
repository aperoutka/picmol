import numpy as np
from scipy.optimize import curve_fit
from sympy import symbols, sympify, diff, preorder_traversal, lambdify
from scipy import constants

from ..conversions import mol2vol
from .cem import PointDisc


def add_zeros(arr1d, num_comp):
  new_arr = np.zeros(arr1d.size + num_comp)
  new_arr[:len(arr1d)] = arr1d
  return new_arr

def add_pc(z_arr, num_comp):
  new_arr = np.zeros((z_arr.shape[0] + num_comp, z_arr.shape[1]))
  new_arr[:z_arr.shape[0],:] = z_arr
  pure_comp_arr = []
  for i in range(num_comp):
    a = np.zeros(num_comp)
    a[i] = 1
    pure_comp_arr += [a]
  new_arr[z_arr.shape[0]:,:] = np.array(pure_comp_arr)
  return new_arr


def get_symbols_sympy_func(expression, xvar: str = 'x'):
  symbols_set = set()
  for arg in preorder_traversal(expression):
    if arg.is_Symbol:
      symbols_set.add(str(arg))
  symbols_list = list(symbols_set)
  x_symbols = sorted([s for s in symbols_list if s.startswith(xvar)])
  other_symbols = sorted([s for s in symbols_list if not s.startswith(xvar)])
  return x_symbols + other_symbols


class QuarticModel:
  def __init__(self, z_data, Hmix, Sex, molar_vol, z=None):
    self.V0 = molar_vol
    self.rec_steps = 10
    self.Rc = constants.R / 1000
    self.num_comp = len(molar_vol)

    self.Hmix_vals = add_zeros(Hmix, self.num_comp)
    self.Sex_vals = add_zeros(Sex, self.num_comp)
    self.z_data = add_pc(z_data, self.num_comp)

    # get mol and vol arrays
    num_pts = {2:10, 3:7, 4:6, 5:5, 6:4}
    self.rec_steps = num_pts[self.num_comp]

    if z is not None:
      self.z = z
    else:
      point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.rec_steps, load=True, store=False)
      self.z = point_disc.points_mfr 

    self.v = mol2vol(self.z, self.V0)

    # fit quartic to Hmix and Sex
    self.fit_Hmix()
    self.fit_Sex()



  @property
  def _polynomial_func(self):
    return self.create_polynomial_for_fitting(self.num_comp)

  @property
  def _df_dx(self):
    return self.df_dx_func(xvar='x')
  
  @property
  def _d2f_dx2(self):
    return self.d2f_dx2_func(xvar='x')

  @property
  def _dMf_dmi(self):
    return self.df_dx_func(xvar='m')

  def df_dx_func(self, xvar: str = 'x'):
    # create function from num_components
    py_func = self.create_polynomial_for_sympy(num_comp=self.num_comp, xvar=xvar)
    # convert to sympy function
    sympy_func = self.python_function_to_sympy(python_function=py_func, xvar=xvar)
    # setup variables
    x_symbols = [symbols(f"{xvar}{i}") for i in range(1,self.num_comp+1)]
    if xvar == 'x':
      df_dxs = {i: None for i in range(self.num_comp-1)}
    else:
      df_dxs = {i: None for i in range(self.num_comp)}
    # get list of arguments
    arg_list = get_symbols_sympy_func(sympy_func, xvar)
    for i in df_dxs.keys():
      df_dx_sympy = diff(sympy_func, x_symbols[i])
      df_dxs[i] = lambdify(arg_list, df_dx_sympy, 'numpy')
    return df_dxs

  def d2f_dx2_func(self, xvar: str = 'x'):
    # create function from num_components
    py_func = self.create_polynomial_for_sympy(num_comp=self.num_comp, xvar=xvar)
    # convert to sympy function
    sympy_func = self.python_function_to_sympy(python_function=py_func, xvar=xvar)
    # setup variables
    x_symbols = [symbols(f"{xvar}{i}") for i in range(1,self.num_comp+1)]
    if xvar == 'x':
      d2f_dx2s = {i: {j: None for j in range(self.num_comp-1)} for i in range(self.num_comp-1)}
    else:
      d2f_dx2s = {i: {j: None for j in range(self.num_comp)} for i in range(self.num_comp)}
    # get list of arguments
    arg_list = get_symbols_sympy_func(sympy_func, xvar)
    for i in d2f_dx2s.keys():
      df_dx_sympy = diff(sympy_func, x_symbols[i])
      for j in d2f_dx2s[i].keys():
        d2f_dx2_sympy = diff(df_dx_sympy, x_symbols[j])
        d2f_dx2s[i][j] = lambdify(arg_list, d2f_dx2_sympy, 'numpy')
    return d2f_dx2s
  
  def python_function_to_sympy(self, python_function, xvar: str = 'x'):
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
    fit, pcov = curve_fit(self._polynomial_func, self.z_data, self.Hmix_vals)
    self._Hmix_params = fit
  
  def fit_Sex(self):
    fit, pcov = curve_fit(self._polynomial_func, self.z_data, self.Sex_vals)
    self._Sex_params = fit

  @property
  def x_values(self):
    return [self.z[:,i] for i in range(self.num_comp-1)]
  
  @property
  def m_values(self):
    return [self.z[:,i] for i in range(self.num_comp)]

  @property
  def Hmix(self):
    return self._polynomial_func(self.z, *self._Hmix_params)

  @property
  def Sex(self):
    return self._polynomial_func(self.z, *self._Sex_params)

  @property
  def Gid_symbols(self):
    return get_symbols_sympy_func(self.Gid_sympy, xvar='x')

  @property
  def Gid_sympy(self):
    xsum = f"(1"
    for i in range(1, self.num_comp):
      xsum += f"-x{i}"
    xsum += ")"

    vbar = "("
    for i in range(1, self.num_comp):
      vbar += f"x{i}*v{i} + "
    vbar += xsum + f"*v{self.num_comp})"

    G = ""
    for i in range(1, self.num_comp):
      G += f"x{i} * ln(x{i}*v{i}/{vbar}) + "
    G += f"{xsum} * ln({xsum}*v{self.num_comp}/{vbar})"
    return sympify(G)

  def Gid(self, T):
    Gid_py = lambdify(self.Gid_symbols, self.Gid_sympy, 'numpy')
    Gid_calc = self.Rc * T * Gid_py(*self.x_values, *self.V0)
    return Gid_calc

  @property
  def dGid_sympy(self):
    df_dx = {i: 0 for i in range(self.num_comp-1)}
    for i in range(self.num_comp-1):
      df_dx[i] = diff(self.Gid_sympy, self.Gid_symbols[i])
    return df_dx

  def dGid(self, T):
    df_dx = np.zeros((self.z.shape[0], self.num_comp-1))
    for i in range(self.num_comp-1):
      df_dx_py = lambdify(self.Gid_symbols, self.dGid_sympy[i], 'numpy')
      df_dx[:,i] = self.Rc * T * df_dx_py(*self.x_values, *self.V0)
    return df_dx

  @property
  def d2Gid_sympy(self):
    d2f_dx2 = {i: {j: 0 for j in range(self.num_comp-1)} for i in range(self.num_comp-1)}
    for i in range(self.num_comp-1):
      df_dx_sympy = self.dGid_sympy[i]
      for j in range(self.num_comp-1):
        d2f_dx2[i][j] = diff(df_dx_sympy, self.Gid_symbols[j])
    return d2f_dx2

  def d2Gid(self, T):
    d2f_dx2 = np.zeros((self.z.shape[0], self.num_comp-1, self.num_comp-1))
    for i in range(self.num_comp-1):
      for j in range(self.num_comp-1):
        d2f_dx2_py = lambdify(self.Gid_symbols, self.d2Gid_sympy[i][j], 'numpy')
        d2f_dx2[:,i,j] = self.Rc * T * d2f_dx2_py(*self.x_values, *self.V0)
    return d2f_dx2

  def Hmix_func(self, T):
    return np.nan_to_num(self.Hmix, nan=0)

  def nTSex_func(self, T):
    return np.nan_to_num(- T * self.Sex, nan=0)

  def Smix_func(self, T):
    s = self.Gid(T) - T * self.Sex
    s = np.nan_to_num(s, nan=0)
    return s

  def ln_gammas(self, T):
    lng_arr = np.zeros((self.z.shape[0], self.num_comp))
    for i in range(self.num_comp):
      lng_arr[:,i] = self._dMf_dmi[i](*self.m_values, *self._Hmix_params) - T * self._dMf_dmi[i](*self.m_values, *self._Sex_params)
    lng_arr /= self.Rc * T
    return lng_arr
  
  def gammas(self, T):
    return np.exp(self.ln_gammas(T))

  def GE(self, T):
    ge = self.Hmix_func(T) + self.nTSex_func(T)
    ge = np.nan_to_num(ge, nan=0)
    return ge

  def GM(self, T):  
    gm = self.Hmix_func(T) + self.Smix_func(T)
    gm = np.nan_to_num(gm, nan=0)
    return gm

  def dGM_dxs(self, T):
    dgm_arr = np.zeros((self.z.shape[0], self.num_comp-1))
    for i in range(self.num_comp-1):
      dgm = self._df_dx[i](*self.x_values, *self._Hmix_params) - T * self._df_dx[i](*self.x_values, *self._Sex_params) + self.dGid(T)[:,i]
      dgm = np.nan_to_num(dgm, nan=0)
      dgm_arr[:,i] = dgm
    return dgm_arr

  def d2GM_dx2s(self, T):
    d2gm_arr = np.zeros((self.z.shape[0], self.num_comp-1, self.num_comp-1))
    for i in range(self.num_comp-1):
      for j in range(self.num_comp-1):
        d2gm = self._d2f_dx2[i][j](*self.x_values, *self._Hmix_params) - T * self._d2f_dx2[i][j](*self.x_values, *self._Sex_params) + self.d2Gid(T)[:,i,j]
        d2gm = np.nan_to_num(d2gm, nan=0)
        d2gm_arr[:,i,j] = d2gm
    return d2gm_arr

  def det_Hij(self, T):
    return np.linalg.det(self.d2GM_dx2s(T))


  @staticmethod
  def create_polynomial_for_fitting(num_comp):
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
  def create_polynomial_for_sympy(num_comp: int, xvar: str = 'x'):
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