import scipy.constants as constants
import math
import numpy as np
import pandas as pd
from pathlib import Path

from .fragmentation import FragmentMolecule


class PermittivityParameters:

	def __init__(self, molecule=None):

			if type(molecule) == str:
				obj = FragmentMolecule(molecule)
				groups = obj.groups  
				self.from_group_contribution_method(groups)
			elif type(molecule) == dict:
				self.from_group_contribution_method(molecule)
			else:
				raise Exception("Could not create permittivity parameters.")
			
			# Check if parameters are nan and set them 0 in this case
			if np.isnan(self.a11_mu2):
					self.a11_mu2 = 0
			if np.isnan(self.a12_alpha):
					self.a12_alpha = 0
			if np.isnan(self.a2):
					self.a2 = 0

	def from_group_contribution_method(self, molecule):

			""" Create relative static permittivity parameters from group contribution method."""

			# Read GC parameter database
			gcparameters = pd.read_csv(Path(__file__).parent / "GCPermittivityParameters.csv", index_col="group", comment="#") 
			all_groups = gcparameters.index.to_list()

			# check if all groups in molecule occur in gc_parameters
			for g in molecule.keys():
				if g not in all_groups:
					raise Exception("Group " + g + " not in GC database.")
					
			self.a11_mu2 = np.sum([molecule[g] * gcparameters.loc[g, "a11_mu2"] for g in molecule.keys()])
			self.a2 = np.sum([molecule[g] * gcparameters.loc[g, "a2"] for g in molecule.keys()])
			self.a12_alpha = np.sum([molecule[g] * gcparameters.loc[g, "a12_alpha"] for g in molecule.keys()])


