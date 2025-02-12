import numpy as np
import pandas as pd
from pathlib import Path
import math, os, sys
from rdkit import Chem
from scipy.interpolate import interp1d
import scipy.optimize as sco
from copy import copy

from .models import FH, NRTL, UNIQUAC, UNIFAC
from .get_molecular_properties import load_molecular_properties, search_molecule
from .models.cem import CEM
from .conversions import mol2vol

from scipy.constants import N_A

def spinodal_fn(z, Hij):
	'''gets spinodal from roots of second derivative'''
	sign_changes = np.diff(np.sign(Hij))  # diff of signs between consecutive elements
	spin_inds = [s for s, sc in enumerate(sign_changes) if sc != 0 and ~np.isnan([sc])]
	if len(spin_inds) > 0:
		return z[spin_inds,:]
	else:
		return None
	
def binodal_fn(smiles, T, gE_model_name: str, thermo_model):
	bi_obj = CEM(smiles=smiles, T=T, gE_model_name=gE_model_name, thermo_model=thermo_model)
	return bi_obj.binodal_matrix_molfrac

def mkdr(dir_path):
	if os.path.exists(dir_path) == False:
		os.mkdir(dir_path)
	return dir_path

def GM_from_x(x, z1, GM):
	"""" interpolate GM given a mol fraction """
	iiall = ~np.isnan(GM)
	return np.interp(x, z1[iiall], GM[iiall])

def Tc_search(Tmax, smiles: list):
	T = Tmax
	dT = 10
	while T > 0:
		model = UNIFAC(T=T, smiles=smiles) # for unifac model
		d2GM = model.det_Hij()
		sign_changes = np.diff(np.sign(d2GM))  # diff of signs between consecutive elements
		spin_inds = [s for s, sc in enumerate(sign_changes) if sc != 0 and ~np.isnan([sc])]
		if len(spin_inds) > 0 and dT == 10:
			if T == Tmax:
				return T
			T += dT-1
			dT = 1
		elif len(spin_inds) > 0 and dT == 1:
			return T
		else:
			T -= dT
	return np.nan



class ThermoModel:

	def __init__(
			self, 
			model_name: str, # thermo model
			KBIModel = None, # option to feed in kbi model
			# if kbi model not specified, the following parameters are required:
			identifiers: list = None, identifier_type: str = None, save_dir: str = None,
			IP=None, # NRTL, UNIQUAC
			phi=None, Smix=None, Hmix=None, # FH
			# optional tuning parameters for temp dependence
			Tmin=100, Tmax=400, dT=5
		):
			
		"""
		Different parameters are required for different thermodynamic models:
		if kbi model fed, the IP/identifiers, etc are not needed
		FIT FROM SIMULATION:
			- UNIQUAC: smiles, IP
			- NRTL: smiles, IP
			- FH: smiles, phi, Gmix, Hmix
		PREDICTIVE:
			- UNIFAC: smiles
		"""

		if KBIModel is not None:
			self.identifiers = KBIModel.unique_mols
			self.identifier_type = "mol_id"
			self.save_dir = mkdr(f"{KBIModel.kbi_method_dir}/{model_name}/")
			# get interaction parameters
			IP_map = {"fh": None, "nrtl": KBIModel.nrtl_taus, "uniquac": KBIModel.uniquac_du, "unifac": None}
			self.IP = IP_map[model_name]
			# load properties for FH analysis
			Smix = KBIModel.Smix
			Hmix = KBIModel.Hmix
			phi = KBIModel.v[:,0]

		else:
			if "smile" not in identifier_type:
				self.identifiers = [identifier.upper() for identifier in identifiers]
			else:
				self.identifiers = identifiers
			self.identifier_type = identifier_type
			self.IP = IP
			self.save_dir = save_dir

		# initialize temperatures
		self.Tmin = Tmin
		self.Tmax = Tmax
		self.dT = dT

		# get type of thermo model
		model_map = {"fh": FH, "nrtl": NRTL, "uniquac": UNIQUAC, "unifac": UNIFAC}
		self.model_name = model_name.lower()
		self.model_type = model_map[self.model_name]
		# initialize thermodynamic model object
		if self.model_type != UNIFAC:
			self.model = self.model_type(smiles=self.smiles, IP=self.IP)
		else: 
			self.model = self.model_type(T=Tmax, smiles=self.smiles)

		# get volume fraction of composition
		self.z = self.model.z
		self.v = mol2vol(self.z, self.molar_vol)

		# for Flory-Huggins model, we need extra parameters for scaling mixing free energy with temperature
		if self.model_type == FH:
			self.model.load_thermo_data(phi=phi, Smix=Smix, Hmix=Hmix)

		''' todo!!! 
		# make binary_temperature_scaling function extendable to multicomponents
		'''

	@property
	def num_comp(self):
		return len(self.identifiers)

	@property
	def T_values(self):
		return np.arange(self.Tmin, self.Tmax+1E-3, self.dT)[::-1]

	@property
	def mol_by_identifier(self):
		# first search for molecules
		for mol in self.identifiers:
			search_molecule(mol, self.identifier_type)
		# then get properties
		mol_props = load_molecular_properties(self.identifier_type)
		mol_by_identifier = mol_props.loc[self.identifiers, :].reset_index()
		return mol_by_identifier

	@property
	def mol_id(self):
		return self.mol_by_identifier["mol_id"].to_numpy()

	@property
	def molar_vol(self):
		return self.mol_by_identifier["molar_vol"].to_numpy()

	@property
	def n_electrons(self):
		return self.mol_by_identifier["n_electrons"].to_numpy()

	@property
	def smiles(self):
		return self.mol_by_identifier["smiles"].to_numpy()
	
	@property
	def mol_name(self):
		return self.mol_by_identifier["mol_name"].to_numpy()
	
	@property
	def mol_class(self):
		return self.mol_by_identifier["mol_class"].to_numpy()
	

	def critical_point(self, spins, T):
		if self.model_type == FH:
			self.xc = self.model.xc
			self.phic = self.model.phic
		else:
			self.xc = np.mean(spins)
			self.phic = mol2vol(([self.xc, 1-self.xc]), self.molar_vol)
		self.Tc = T

	
	def assign_solute_mol(self):
		solute_mol = None
		# first check for extractant
		for i, mol in enumerate(self.mol_id):
			if self.mol_class[i] == 'extractant':
				solute_mol = mol
		# then check for modifier
		if solute_mol is None:
			for i, mol in enumerate(self.mol_id):
				if self.mol_class[i] == 'modifier':
					solute_mol = mol
		# then check for solute
		if solute_mol is None:
			for i, mol in enumerate(self.mol_id):
				if self.mol_class[i] == 'solute':
					solute_mol = mol
		# then solvent
		if solute_mol is None:
			for i, mol in enumerate(self.mol_id):
				if self.mol_class[i] == 'solvent':
					solute_mol = mol

		self.solute_mol = solute_mol
		return self.solute_mol
	
	@property
	def solute(self):
		try:
			self.solute_mol
		except AttributeError: 
			self.assign_solute_mol()
		return self.solute_mol

	@property
	def solute_loc(self):
		return np.where(self.mol_id==self.solute)[0][0]
	
	@property
	def solute_name(self):
		return self.mol_name[self.solute_loc]

	
	def binary_binodal_fn(self, sp1, sp2, GM, dGM):
		'''minimize binodal function'''
		
		iiall = ~np.isnan(dGM) & ~np.isnan(GM)
		ii2 = (self.z[:,self.solute_loc] > sp2) & ~np.isnan(dGM) & ~np.isnan(GM)

		Gx_from_x = interp1d(self.z[:,0][iiall], GM[iiall], kind='cubic', fill_value="extrapolate")

		def objective(vars):
			'''calculates binodal pts for a binary mixture'''
			global x1, x2, dGx, dGx2
			x1, x2 = vars
			# mixing free energy
			Gx = Gx_from_x(x1)
			Gx2 = Gx_from_x(x2)
			# Derivatives
			h = 1e-5
			dGx = (Gx_from_x(x1 + h) - Gx_from_x(x1 - h)) / (2 * h)
			dGx2 = (Gx_from_x(x2 + h) - Gx_from_x(x2 - h)) / (2 * h)
			# binodal definitions
			eq1 = dGx2 - dGx
			eq2 = dGx - (Gx2 - Gx)/(x2 - x1)
			eq3 = dGx2 - (Gx2 - Gx)/(x2 - x1)	
			# return sum of eqns
			return eq1 + eq2 + eq3
		
		# minimize the binodal funciton to solve for binodal pts
		bis = sco.minimize(objective, x0=[0.01,0.9], bounds=((1E-5,sp1),(sp2,0.9999)))

		bi1 = bis.x[0]
		bi2 = np.interp(dGx, dGM[ii2],  self.z[:,self.solute_loc][ii2])
		return [bi1, bi2]
	
	def run_binary(self):
		GM_arr = np.zeros((len(self.T_values), self.z.shape[0]))
		d2GM_arr = np.zeros((len(self.T_values), self.z.shape[0]))
		sp_matrix = np.zeros((len(self.T_values), self.num_comp))
		bi_matrix = np.zeros((len(self.T_values), self.num_comp))
		GM_sp_matrix = np.zeros((len(self.T_values), self.num_comp))
		GM_bi_matrix = np.zeros((len(self.T_values), self.num_comp))
		v_sp_matrix = np.zeros((len(self.T_values), self.num_comp))
		v_bi_matrix = np.zeros((len(self.T_values), self.num_comp))
		_Tc_found = False
		
		for t, T in enumerate(self.T_values):
			# self.model = self.model_type(smiles=self.smiles, IP=self.IP, T=T)
			# set variables to nan
			sp1, sp2, bi1, bi2 = np.empty(4)*np.nan

			# for just FH model
			if self.model_type == FH:
				GM_arr[t,:] = self.model.GM(self.v[:,0], T)
				d2GM_arr[t,:] = self.model.det_Hij(self.v[:,0],T)
				try:
					sp1, sp2, bi1, bi2 = self.model.ps_calc(self.model.phic, T)
					if _Tc_found == False and np.all(~np.isnan([sp1,sp2])):
						self.critical_point([sp1, sp2], T)
						_Tc_found = True
				except:
					pass

			# unifac, requires a new model object at each temperature
			elif self.model_type == UNIFAC:
				self.model = self.model_type(T=T, smiles=self.smiles)
				GM_arr[t,:] = self.model.GM()
				d2GM_arr[t,:] = self.model.det_Hij()
				sp_vals = spinodal_fn(self.z, d2GM_arr[t,:])
				if sp_vals is not None:
					sp1, sp2 = [min(sp_vals[:,self.solute_loc]), max(sp_vals[:,self.solute_loc])]
					bi_vals = binodal_fn(smiles=self.smiles, T=T, gE_model_name="unifac", thermo_model=UNIFAC)
					bi_vals = bi_vals[:,:,self.solute_loc]
					bi1, bi2 = bi_vals.min(), bi_vals.max()
					# search for critical point
					if _Tc_found == False:
						self.critical_point([sp1, sp2], T)
						_Tc_found = True
						
			# for uniquac and nrtl
			elif (self.model_type == UNIQUAC) or (self.model_type == NRTL): 
				GM_arr[t,:] = self.model.GM(T)
				d2GM_arr[t,:] = self.model.det_Hij(T)
				sp_vals = spinodal_fn(self.z, d2GM_arr[t,:])
				if sp_vals is not None:			
					sp1, sp2 = [min(sp_vals[:,self.solute_loc]), max(sp_vals[:,self.solute_loc])]
					bi_vals = binodal_fn(smiles=self.smiles, T=T, gE_model_name="uniquac", thermo_model=self.model_type)
					bi_vals = bi_vals[:,:,self.solute_loc]
					bi1, bi2 = bi_vals.min(), bi_vals.max()
					# search for critical point
					if _Tc_found == False:
						self.critical_point([sp1, sp2], T)
						_Tc_found = True

			sp_matrix[t,:] = [sp1, sp2]
			bi_matrix[t,:] = [bi1, bi2]
			sp1_ind = np.abs(self.z[:,self.solute_loc] - sp1).argmin()
			sp2_ind = np.abs(self.z[:,self.solute_loc] - sp2).argmin()
			bi1_ind = np.abs(self.z[:,self.solute_loc] - bi1).argmin()
			bi2_ind = np.abs(self.z[:,self.solute_loc] - bi2).argmin()

			if np.all(~np.isnan([sp1, sp2, bi1, bi2])):
				v_sp_matrix[t,:] = [self.v[sp1_ind, self.solute_loc], self.v[sp2_ind, self.solute_loc]]
				v_bi_matrix[t,:] = [self.v[bi1_ind, self.solute_loc], self.v[bi2_ind, self.solute_loc]]
				GM_sp_matrix[t,:] = [GM_arr[t,sp1_ind], GM_arr[t,sp2_ind]]
				GM_bi_matrix[t,:] = [GM_arr[t,bi1_ind], GM_arr[t,bi2_ind]]
			else:
				v_sp_matrix[t,:] = [np.nan, np.nan]
				v_bi_matrix[t,:] = [np.nan, np.nan]
				GM_sp_matrix[t,:] = [np.nan, np.nan]
				GM_bi_matrix[t,:] = [np.nan, np.nan]
				

		self.GM = GM_arr
		self.d2GM = d2GM_arr
		self.x_sp = sp_matrix
		self.v_sp = v_sp_matrix
		self.x_bi = bi_matrix
		self.v_bi = v_bi_matrix
		self.GM_sp = GM_sp_matrix
		self.GM_bi = GM_bi_matrix

		# creates pd.DataFrame obj and save
		if self.save_dir is not None:
			df = pd.DataFrame(index=self.T_values)
			df.index.name = "T"
			for i in range(self.num_comp):
				df[f"x_sp{i+1}"] = self.x_sp[:,i]
				df[f"v_sp{i+1}"] = self.v_sp[:,i]
				df[f"x_bi{i+1}"] = self.x_bi[:,i]
				df[f"v_bi{i+1}"] = self.v_bi[:,i]
			df.to_csv(f"{self.save_dir}{self.model_name}_phase_instability_values.csv", index=True)


	def calculate_saxs_Io(self):
		R = 8.314E-3 # kJ/mol
		re2 = 7.9524E-26 # cm^2
		kb = R/N_A

		try:
			self.d2GM
		except AttributeError:
			self.run_binary_temperature_scaling()

		V_bar_i0 = self.z @ self.molar_vol
		N_bar = self.z @ self.n_electrons
		Ni = self.n_electrons

		drho_dx = np.zeros(self.z.shape[0])
		for j in range(self.num_comp-1):
			if self.num_comp > 2:
				drho_dx += N_A*((V_bar_i0*(Ni[j]-Ni[-1]) - N_bar*(V_bar_i0[j]-V_bar_i0[-1]))) / ((V_bar_i0)**2)
			else:
				drho_dx += N_A*((Ni[j]*self.molar_vol[-1]) - (Ni[-1]*self.molar_vol[j])) / ((V_bar_i0)**2)


		I0_arr = np.empty((len(self.T_values), self.z.shape[0]))
		x_I0_max = np.empty((len(self.T_values), self.num_comp))
		v_I0_max = np.empty((len(self.T_values), self.num_comp))
		df_I0_max = pd.DataFrame(columns=['T', 'I0'])
		df_I0 = pd.DataFrame()

		for t, T in enumerate(self.T_values):
			S0 = kb * T * V_bar_i0 / self.d2GM[t]
			I0 = re2 * (drho_dx**2) * S0
			I0_arr[t] = I0
			I0 = np.array([np.round(i, 4) for i in I0])
			I0_filter = ~np.isnan(I0)
			if len(I0_filter) > 0:
				I0_max = max(I0[I0_filter])
				v_I0_max[t] = self.v[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
				x_I0_max[t] = self.z[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
				try:
					if T > self.Tc:
						df_I0[T] = I0
						if T % 10 == 0:
							new_row = {"T": T, "I0": I0_max}
							df_I0_max = df_I0_max._append(new_row, ignore_index=True)
				except:
					pass
			
		self.I0_max = df_I0_max
		self.I0_arr = I0_arr
		self.x_I0_max = x_I0_max
		self.v_I0_max = v_I0_max

		# convert to pandas df for saving
		if self.save_dir is not None:
			df_I0_max.to_csv(f"{self.save_dir}{self.model_name}_Io_max_values.csv", index=False)
			df_I0.to_csv(f"{self.save_dir}{self.model_name}_Io_values.csv", index=False)

	def run_ternary(self):

		all_GMs = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		all_d2GMs = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		all_sp_ls = []
		all_bi_ls = []

		_Tc_found = False

		for t, T in enumerate(self.T_values):

			model = UNIFAC(T=T, smiles=self.smiles)
			GM = model.GM()
			all_GMs[t] = GM

			det_Hij = model.det_Hij()
			all_d2GMs[t] = det_Hij

			# filter 2nd derivative for spinodal determination
			mask = (all_d2GMs[t] > -1) & (all_d2GMs[t] <= 1)
			sps = np.array(model.z[mask,:])
			all_sp_ls += [sps]
			try:
				bi_vals = binodal_fn(smiles=self.smiles, T=T, gE_model_name="unifac", thermo_model=UNIFAC)
				all_bi_ls += [bi_vals]
			except:
				pass

			if len(model.z[mask,:]) > 0 and _Tc_found == False:
				self.Tc = T
				_Tc_found = True

		self.d2GM = all_d2GMs
		self.GM = all_GMs
		self.x_sp = all_sp_ls
		self.x_bi = all_bi_ls

