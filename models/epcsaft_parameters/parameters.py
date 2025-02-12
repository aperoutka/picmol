from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
import sys as sys_arg
from scipy import constants

from .molecular_fingerprints import compute_morgan_fingerprints
from .rdkit_properties import get_rdkit_mol_properties
from .permittivity import PermittivityParameters


class EPCSAFTParameters:
	""" defining parameters for EPC-SAFT thermodynamic model """
	
	def __init__(self, smiles):
		self.smiles_list = smiles
		
		# load ML-SAFT model
		with open(Path(__file__).parent / "mlsaft_model.pkl", "rb") as f:
			self.model = load(f)
		

	@property
	def morgan_fp(self):
		return compute_morgan_fingerprints(self.smiles_list)

	@property
	def m(self):
		try: 
			self.df_mlsaft
		except AttributeError:
			self.make_predictions()
		return self.df_mlsaft["m"].to_numpy()
	
	@property
	def sigma(self):
		try: 
			self.df_mlsaft
		except AttributeError:
			self.make_predictions()
		return self.df_mlsaft["sigma"].to_numpy()
	
	@property
	def epsilon_k(self):
		try: 
			self.df_mlsaft
		except AttributeError:
			self.make_predictions()
		return self.df_mlsaft["epsilon_k"].to_numpy()

	@property
	def epsilon_k_J(self):
		''' epsilon K in Joules '''
		return self.epsilon_k / constants.N_A

	@property
	def epsilon_k_K(self):
		''' epsilon K in Kelvin '''
		return self.epsilon_k / constants.R
	
	@property
	def epsilon_k_ab(self):
		try: 
			self.df_mlsaft
		except AttributeError:
			self.make_predictions()
		return self.df_mlsaft["epsilon_k_ab"].to_numpy()
	
	@property
	def kappa_ab(self):
		try: 
			self.df_mlsaft
		except AttributeError:
			self.make_predictions()
		return self.df_mlsaft["kappa_ab"].to_numpy()

	def make_predictions(self):
		target_columns = ["m", "sigma", "epsilon_k", "epsilonAB", "KAB"]
		preds = self.model.predict(self.morgan_fp)
		df = pd.DataFrame(preds, columns=target_columns)
		df.columns = ["m", "sigma", "epsilon_k", "epsilon_k_ab", "kappa_ab"]
		self.df_mlsaft = df

	@property
	def n_hb_acceptors(self):
		try: 
			self.mol_properties
		except AttributeError:
			self.get_mol_properties()
		return self.mol_properties["n_hb_acceptors"]

	@property
	def n_hb_donors(self):
		try: 
			self.mol_properties
		except AttributeError:
			self.get_mol_properties()
		return self.mol_properties["n_hb_donors"]
	
	@property
	def n_hb_sites_pure(self):
		try: 
			self.mol_properties
		except AttributeError:
			self.get_mol_properties()
		return np.array([int(n_site) for n_site in self.mol_properties["n_hb_sites"]])
	
	@property
	def n_hb_sites(self):
		return int(sum(self.n_hb_sites_pure))
	
	@property
	def hb_types(self):
		return self.mol_properties["hb_types"]
	
	@property
	def q(self):
		try: 
			self.mol_properties
		except AttributeError:
			self.get_mol_properties()
		return self.mol_properties["q"]
	
	@property
	def Vm_pure(self):
		try: 
			self.mol_properties
		except AttributeError:
			self.get_mol_properties()
		return self.mol_properties["Vm"]
	
	@property
	def Vm_pure_m3mol(self):
		'''convert molar volume from nm^3/molecule to m^3/mol'''
		return self.Vm_pure * constants.N_A / (1E30)

	@property
	def rho_pure(self):
		return 1/self.Vm_pure
	
	@property
	def mw(self):
		try: 
			self.mol_properties
		except AttributeError:
			self.get_mol_properties()
		return self.mol_properties["mw"]

	def get_mol_properties(self):
		self.mol_properties = get_rdkit_mol_properties(self.smiles_list)

	@property
	def a11_mu2(self):
		try: 
			self.params_list
		except AttributeError:
			self.permattivity_parameters()
		return np.array([params.a11_mu2 for params in self.params_list])
	
	@property
	def a12_alpha(self):
		try: 
			self.params_list
		except AttributeError:
			self.permattivity_parameters()
		return np.array([params.a12_alpha for params in self.params_list])
	
	@property
	def a2(self):
		try: 
			self.params_list
		except AttributeError:
			self.permattivity_parameters()
		return np.array([params.a2 for params in self.params_list])

	def permattivity_parameters(self):
		self.params_list = [PermittivityParameters(smile) for smile in self.smiles_list]
		
	@property
	def sigma_born(self):
		'''for now, just assume sigma_born = sigma; typically ionic radii'''
		return self.sigma