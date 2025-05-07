import numpy as np

def mol2vol(val, V0):
  r"""
  Convert mol frac to vol frac.

  .. math::
    \phi_i = \frac{x_i V_i}{\sum_j x_j V_j}

  where:
  
    * :math:`\phi_i` is the volume fraction of component :math:`i`
    * :math:`x_i` is the mole fraction of component :math:`i`
    * :math:`V_i` is the molar volume of component :math:`i`

  :param val: mol fraction
  :type val: float or numpy.ndarray
  :param V0: molar volume
  :type V0: numpy.ndarray
  :return: volume fraction
  :rtype: float or numpy.ndarray
  """
  V0 = np.array(V0)
  # if val is a float, convert to array
  if type(val) in [float, np.float64, np.float32]:
    z_arr = np.array([val, 1-val])
    vbar = z_arr @ V0
    v_arr = z_arr * V0 / vbar
    return float(v_arr[0])
  else:
    z_arr = np.array(val)
    vbar = z_arr @ V0
    v_arr = z_arr * V0 / vbar[:,np.newaxis]
    return v_arr

def vol2mol(val, V0):
  r"""
  Convert vol frac to mol frac.

  .. math::
    x_i = \frac{\frac{\phi_i}{V_i}}{\sum_j \frac{\phi_j}{V_j}}

  where:

    * :math:`x_i` is the mole fraction of component :math:`i`
    * :math:`\phi_i` is the volume fraction of component :math:`i`
    * :math:`V_i` is the molar volume of component :math:`i`

  :param val: volume fraction
  :type val: float or numpy.ndarray
  :param V0: molar volume
  :type V0: numpy.ndarray
  :return: mol fraction
  :rtype: float or numpy.ndarray
  """
  V0 = np.array(V0)
  # if val is a float, convert to array
  if type(val) in [float, np.float64, np.float32]:
    v_arr = np.array([val, 1-val])
    vfrac = v_arr / V0
    z_arr = vfrac / vfrac.sum()
    return float(z_arr[0])
  else:
    v_arr = np.array(val)
    vfrac = v_arr / V0[np.newaxis,:]
    z_arr = vfrac / (vfrac.sum(axis=1)[:,np.newaxis])
    return z_arr


