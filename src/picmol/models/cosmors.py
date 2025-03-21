import numpy as np
import pandas as pd
from pathlib import Path
import functools

from opencosmorspy import COSMORS
from opencosmorspy.parameterization import openCOSMORS24a
from opencosmorspy.input_parsers import SigmaProfileParser

from .cem import PointDisc

class COSMORSModel:
  """A COSMORS model object that can be used to calculate activity coefficients, chemical potentials, and Gibbs free energies for a given set of molecules"""
  def __init__(self, identifiers, z=None):
    self.identifiers = identifiers
    self.num_comp = len(self.identifiers)
    self.rec_steps = 10-self.num_comp

    if z is not None:
      self.z = z
    else:
      point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.rec_steps, load=True, store=False)
      self.z = point_disc.points_mfr
    self.R = 8.314E-3 # kJ/(mol K)

    # create object and add molecules
    self.crs = COSMORS(par=openCOSMORS24a())
    self.crs.par.calculate_contact_statistics_molecule_properties = True
    for mol in self.mol_names:
      mol_file = self.get_orcacosmo_file(mol)
      self.crs.add_molecule([mol_file])


  @property
  def mol_names(self):
    property_file = Path(__file__).parent.parent / "data" / "molecular_properties.csv"
    property_df = pd.read_csv(property_file)
    for col in property_df:
      try:
        mols = property_df.set_index(col).loc[self.identifiers, "cosmo_name"]
        return mols
      except:
        pass

  @staticmethod
  def get_orcacosmo_file(mol_name):
    return Path(__file__).parent / "cosmors_molecules" / f"{mol_name}" / "COSMO_TZVPD" / f"{mol_name}_c000.orcacosmo"

  def sigma_profile_parser(self, mol_name):
    """Return the sigma profile parser for a given molecule"""
    # idx = np.where(self.mol_names == mol_name)[0][0]
    idx = self.mol_names.index(mol_name) ## this isn't working, check what values it's returning!!!
    parser = SigmaProfileParser(self.crs.enth.mol_lst[idx].cosmo_struct_lst[0].filepath)
    # sigmas, areas = parser.cluster_and_get_sigma_profile()
    parser.calculate_sigma_moments()
    return parser
    
  def molar_volume(self, mol_name):
    """Return the molar volume for a given molecule"""
    return self.sigma_profile_parser(mol_name)['volume']
  
  def sasa(self, mol_name):
    """Return the solvent accessible surface area for a given molecule"""
    return self.sigma_profile_parser(mol_name)['area']

  def dipole_moment(self, mol_name):
    """Return the magnitude of the dipole moment vector for a given molecule"""
    return np.linalg.norm(self.sigma_profile_parser(mol_name)['dipole_moment'])
  
  @functools.lru_cache(maxsize=None, typed=False)
  def _calculate_cosmors(self, T):
    """functools.lru_cache(maxsize=None) is used to cache the results of the function so that it is not recalculated every time it is called with the same arguments, typed=False is used to allow int and float to be treated as the same key"""
    # check that the jobs have been cleared
    self.crs.clear_jobs()
    # add jobs at temperature
    for x in self.z:
      self.crs.add_job(x, T, refst='pure_component')
    # calculate the activity coefficients & energies
    results = self.crs.calculate()
    return results  

  def calculate(self, T):
    return self._calculate_cosmors(T)

  def gammas(self, T):
    """Calculate the activity coefficients at a given temperature"""
    return np.exp(self.ln_gammas(T))

  def ln_gammas(self, T):
    """Calculate the natural log of the activity coefficients at a given temperature"""
    return self.calculate(T)['tot']['lng']
  
  def ln_gammas_enthalpic(self, T):
    """Calculate the natural log of the enthalpic activity coefficients at a given temperature"""
    return self.calculate(T)['enth']['lng']

  def ln_gammas_entropic(self, T): 
    """Calculate the natural log of the entropic activity coefficients at a given temperature"""
    return self.calculate(T)['comb']['lng']

  def mu(self, T):
    """Calculate the chemical potential at a given temperature"""
    return self.R * T * self.ln_gammas(T)

  def GE(self, T):
    """Calculate the excess Gibbs free energy at a given temperature"""
    return self.R * T * np.sum(self.z * self.ln_gammas(T), axis=1)

  def GID(self, T):
    """Ideal Gibbs Free energy at a given temperature"""
    return self.R * T * np.sum(self.z * np.log(self.z), axis=1)
  
  def GM(self, T):
    """Calculate the Gibbs free energy at a given temperature"""
    return self.GE(T) + self.GID(T)

  def dGM_dxs(self, T):
    """for now just use np.gradient -- input actual form later"""
    return np.gradient(self.GM(T), self.z[:,0])

  def det_Hij(self, T):
    """for now just use np.gradient -- input actual form later"""
    return np.gradient(self.dGM_dxs(T), self.z[:,0])  
