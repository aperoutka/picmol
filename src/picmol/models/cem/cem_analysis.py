import os
from pathlib import Path
import copy
import numpy as np

from ..unifac import UNIFAC

from .point_discretization import PointDisc
from .lle import MiscibilityAnalysis

def _make_point_discretizations():
  """
  Create point discretizations if they do not exist.

  This function generates point discretization data for multicomponent systems with
  different recursion steps. The data is stored in a directory named
  "discretization" within the parent directory of the current file. If the
  directory or the individual discretization files do not exist, they are
  created. This function ensures that the necessary discretization files
  are available for subsequent LLE calculations.
  """

  # create point discretizations if necessary
  discretizations_todo = [[2, 8], [2,10], [2,12], [2,14], [3, 7], [3,8], [3,10], [4, 6], [5, 5], [6, 4]]

  discretization_path = os.path.join(Path(__file__).parent.parent, "discretization")
  if not os.path.isdir(discretization_path):
    os.mkdir(discretization_path)

  for todo_el in discretizations_todo:
    filename = os.path.join(discretization_path, str(todo_el[0]) + "_" + str(todo_el[1]))
    if not os.path.isdir(filename):
      PointDisc(num_comp=todo_el[0], recursion_steps=todo_el[1], load=False, store=True)

        
class CEM:
  """
  Calculates Liquid-Liquid Equilibrium (LLE) data.

  This class takes component names and other inputs to set up and execute an
  LLE analysis to determine the binodal curve. The calculated
  LLE data is then stored as class attributes.

  :param num_comp:
      The number of components in the system.
  :type num_comp: int
  :param rec_steps:
      The number of recursion steps used in the point discretization.
  :type rec_steps: int
  :param G_mix:
      Array of Gibbs free energy of mixing as a function of mol fractions in point discretization mol fraction matrix.
  :type G_mix: numpy.ndarray
  :param activity_coefs:
      Array of activity coefficients for each component present.
  :type activity_coefs: numpy.ndarray

  :ivar point_disc:
      A PointDisc object representing the point discretization of the
      composition space.
  :vartype point_disc: PointDisc
  :ivar binodal_matrix_molfrac:
        A numpy array containing the compositions of the phases at
        equilibrium, expressed as mol fractions. The rows correspond to
        unique points on the binodal curve, and the columns correspond to
        the components.
  :vartype binodal_matrix_molfrac: numpy.ndarray
  :ivar binodal_matrix_cartcoords:
        A numpy array containing the compositions of the phases at
        equilibrium, expressed as Cartesian coordinates. The rows
        correspond to unique points on the binodal curve.
  :vartype binodal_matrix_cartcoords: numpy.ndarray
  :ivar binodal_matrix_inds:
        A numpy array containing the indices of the points on the binodal
        curve within the discretized composition space.
  :vartype binodal_matrix_inds: numpy.ndarray
  """
  def __init__(self, num_comp, rec_steps, G_mix, activity_coefs):
    # make point discretizations if they do not exist
    _make_point_discretizations()
      
    self.num_comp = num_comp
    self.point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=rec_steps, load=True, store=False)
    
    # set up the lle analysis
    lle_analysis = MiscibilityAnalysis(self.point_disc, G_mix, activity_coefs, self.num_comp)
    
    # rows = unique pts, cols = components
    self.binodal_matrix_molfrac = lle_analysis.compute_phase_eq_molfrac()
    self.binodal_matrix_cartcoords = lle_analysis.compute_phase_eq_cartcoord()
    self.binodal_matrix_inds = lle_analysis.compute_phase_eq_indices()
