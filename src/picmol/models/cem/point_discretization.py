
import numpy as np
import itertools
import os
from pathlib import Path
import math

class PointDisc:
  """
  Discretizes the composition space of a multi-component system using a simplex.

  This class generates a set of points within the composition space of a
  multi-component system by recursively subdividing a simplex. The points
  can be used to represent the possible compositions of mixtures in the system,
  for example, in phase equilibrium calculations.

  :param num_comp:
      The number of components in the system. Must be greater than or equal to 2.
  :type num_comp: int
  :param recursion_steps:
      The number of recursion steps used to generate the discretization.
      This determines the density of points. Must be greater than 1.
  :type recursion_steps: int
  :param load:
      If True, loads a previously stored discretization from a file.
  :type load: bool
  :param store:
      If True, stores the generated discretization to a file. This is only
      relevant if `load` is False.
  :type store: bool

  :ivar num_comp:
      The number of components in the system.
  :vartype num_comp: int
  :ivar n:
      The dimension of the simplex (number of components minus 1).
  :vartype n: int
  :ivar recursion_steps:
      The number of recursion steps.
  :vartype recursion_steps: int
  :ivar epsilon:
      A small tolerance value used for numerical comparisons.
  :vartype epsilon: float
  :ivar filename:
      The path to the file where the discretization is stored (if `store` is True).
  :vartype filename: str
  :ivar vertices_outer_simplex:
      A list of numpy arrays representing the vertices of the outer simplex
      in Cartesian coordinates.
  :vartype vertices_outer_simplex: list[numpy.ndarray]
  :ivar matrix_mfr_to_cart:
      A matrix used to transform molar fractions (barycentric coordinates)
      to Cartesian coordinates.
  :vartype matrix_mfr_to_cart: numpy.ndarray
  :ivar matrix_cart_to_mfr:
      A matrix used to transform Cartesian coordinates to molar fractions.
  :vartype matrix_cart_to_mfr: numpy.ndarray
  :ivar points_mfr:
      A numpy array containing the discretized points in molar fraction
      (barycentric) coordinates.
  :vartype points_mfr: numpy.ndarray
  :ivar points_cart:
      A numpy array containing the discretized points in Cartesian coordinates.
  :vartype points_cart: numpy.ndarray
  :ivar stepsize:
      The step size used in the discretization, determined by the
      `recursion_steps`.
  :vartype stepsize: float
  """
  def __init__(self, num_comp, recursion_steps, load, store):
    self.num_comp = num_comp
    self.n = self.num_comp - 1  # we need a simplex in R^n to store the surrounding simplex
    self.recursion_steps = recursion_steps
    self.epsilon = 0.0001  # for comparisons
    
    discretization_path = os.path.join(Path(__file__).parent.parent, "discretization")

    self.filename = f"{discretization_path}//" + str(self.num_comp) + "_" + str(self.recursion_steps) + "//" + str(
      self.num_comp) + "_" + str(self.recursion_steps)

    if not os.path.isdir(f"{discretization_path}//" + str(self.num_comp) + "_" + str(self.recursion_steps)):
      os.mkdir(f"{discretization_path}//" + str(self.num_comp) + "_" + str(self.recursion_steps))

    self.vertices_outer_simplex = self.construct_outer_simplex()

    # to get the barycentric coordinates lambda for a point p in R^n, we use the matrix A, where the first row
    # contains only ones and the columns below are given by the vertices of the outer simplex, A * lambda = (1, p)
    self.matrix_mfr_to_cart, self.matrix_cart_to_mfr = self.get_basis_change(self.vertices_outer_simplex)

    self.points_mfr = []
    self.points_cart = []

    self.stepsize = 1 / int(2 ** self.recursion_steps)

    if load:
      self.points_mfr = np.load(self.filename + "_molar_fr_p" + ".npy")
      self.points_cart = np.load(self.filename + "_cart_coords_p" + ".npy")

      sort_inds = self.points_mfr[:, 0].argsort()
      self.points_mfr = self.points_mfr[sort_inds]
      self.points_cart = self.points_cart[sort_inds]

      zero_mask = np.all(self.points_mfr >= 0, axis=1)
      self.points_mfr = self.points_mfr[zero_mask]
      self.points_cart = self.points_cart[zero_mask]

      # correct z, if sum is not 1
      self.points_mfr[:,-1] = 1-self.points_mfr[:,:-1].sum(axis=1)

    else:
      # add pure components as first points
      for v in self.vertices_outer_simplex:
        self.points_cart.append(v)
        mfr = self.transform_cartesian_to_molar_fr(v)
        self.points_mfr.append(mfr)

      self.get_points(base=int(2 ** self.recursion_steps))
      self.points_mfr = np.array(self.points_mfr)

      sort_inds = self.points_mfr[:, 0].argsort()
      self.points_mfr = self.points_mfr[sort_inds]
      self.points_cart = self.points_cart[sort_inds]

      if store:
        np.save(self.filename + "_molar_fr_p", self.points_mfr)
        np.save(self.filename + "_cart_coords_p", self.points_cart)

  def get_points(self, base):
    """
    Generates the discretized points within the simplex.

    This method recursively generates points within the simplex based on the
    number of recursion steps. The points are generated in both molar fraction
    and Cartesian coordinates.

    :param base:
        The base value used for generating the combinations, which is
        2 raised to the power of the number of recursion steps.
    :type base: int
    """
    todo = list(itertools.combinations_with_replacement(list(range(base + 1)), self.num_comp - 1))
    stepsize = 1 / base

    index = 0
    while len(todo) > 0:
      index = index + 1
      combination = todo.pop()
      # if sum is zero, then it is the last pure component
      if 0 < sum(combination) <= base:
        # pures are already added
        if max(combination) != base:
          perm = itertools.permutations(combination)
          for perm_el in set(perm):
            self.points_mfr.append(np.array(list(perm_el) + [base - sum(perm_el)]) * stepsize)
            self.points_cart.append(self.transform_molar_fr_to_cartesian(self.points_mfr[-1]))

  def construct_outer_simplex(self):
    """
    Constructs the outer simplex in Cartesian coordinates.

    This method generates the vertices of a regular n-simplex in R\ :sup:`n`, where
    n is the number of components minus 1. For the case of a 3-component
    system, the simplex is rotated by 285 degrees.

    :returns:
        A list of numpy arrays, where each array represents the Cartesian
        coordinates of a vertex of the simplex.
    :rtype: list[numpy.ndarray]
    """

    # construct a regular n simplex in [0,1]^n
    vertices_outer_simplex = []
    for i in range(self.n):
      basis_vector = np.zeros(self.n)
      basis_vector[i] = 1 / np.sqrt(2)
      vertices_outer_simplex.append(basis_vector)

    # the last point
    vertices_outer_simplex.append(np.ones(self.n) * (1 + np.sqrt(self.n + 1)) / (self.n * np.sqrt(2)))

    if self.num_comp == 3:
      # rotation with psi
      psi = 2 * np.pi * 285 / 360
      rotation_matrix = np.array([[np.cos(psi), -1 * np.sin(psi)], [np.sin(psi), np.cos(psi)]])
      for i in range(len(vertices_outer_simplex)):
        vertices_outer_simplex[i] = np.matmul(rotation_matrix, vertices_outer_simplex[i])

    return vertices_outer_simplex

  def transform_molar_fr_to_cartesian(self, molar_fractions):
    """
    Transforms molar fractions to Cartesian coordinates.

    This method converts a composition represented as molar fractions
    (barycentric coordinates) to Cartesian coordinates using the
    precomputed transformation matrix.

    :param molar_fractions:
        A numpy array representing the composition in molar fractions.
    :type molar_fractions: numpy.ndarray

    :returns:
        A numpy array representing the composition in Cartesian coordinates.
    :rtype: numpy.ndarray
    """
    return np.matmul(self.matrix_mfr_to_cart, molar_fractions)[1:]

  def transform_cartesian_to_molar_fr(self, cartesian_point):
    """
    Transforms Cartesian coordinates to molar fractions.

    This method converts a composition represented as Cartesian coordinates
    to molar fractions (barycentric coordinates) using the precomputed
    transformation matrix.

    :param cartesian_point:
        A numpy array representing the composition in Cartesian coordinates.
    :type cartesian_point: numpy.ndarray

    :returns:
        A numpy array representing the composition in molar fractions.
    :rtype: numpy.ndarray
    """
    vector = np.empty(self.n + 1)
    vector[0] = 1
    vector[1:] = cartesian_point

    return np.matmul(self.matrix_cart_to_mfr, vector)

  @staticmethod
  def euclidean_distance(p1, p2):
    """
    Calculates the Euclidean distance between two points.

    This static method computes the Euclidean distance between two points
    in Cartesian coordinates.

    :param p1:
        A numpy array representing the coordinates of the first point.
    :type p1: numpy.ndarray
    :param p2:
        A numpy array representing the coordinates of the second point.
    :type p2: numpy.ndarray

    :returns:
        The Euclidean distance between the two points.
    :rtype: float
    """
    return np.sqrt(sum(np.square(p1 - p2)))

  @staticmethod
  def get_basis_change(vertices_cartesian):
    """
    Calculates the basis change matrices between molar fractions and Cartesian coordinates.

    This static method computes the matrices A and A_inv, which are used to
    transform between molar fractions (barycentric coordinates) and Cartesian
    coordinates. The matrix A is constructed from the vertices of the simplex.

    :param vertices_cartesian:
        A list of numpy arrays, where each array represents the Cartesian
        coordinates of a vertex of the simplex.
    :type vertices_cartesian: list[numpy.ndarray]

    :returns:
        A tuple containing the matrix A and its inverse A_inv.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    matrix = np.empty((len(vertices_cartesian), len(vertices_cartesian)))
    matrix[0] = np.ones(len(vertices_cartesian))
    for i in range(1, len(vertices_cartesian)):
      for j in range(len(vertices_cartesian)):
        matrix[i][j] = vertices_cartesian[j][i - 1]

    return matrix, np.linalg.inv(matrix)

  @staticmethod
  def volume_simplex(vertices):
    r"""
    Calculates the volume of an n-simplex in R\ :sup:`n`.

    This static method computes the volume of a simplex given its vertices
    using the determinant of a matrix formed from the vertex coordinates.
    
    :param vertices: A list of numpy arrays, where each array represents the Cartesian coordinates of a vertex of the simplex.
    :type vertices: list[numpy.ndarray]

    :returns: The volume of the simplex.
    :rtype: float
    """
    return np.abs(np.linalg.det(vertices[1:] - vertices[0])) / math.factorial(len(vertices) - 1)
