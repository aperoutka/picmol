import numpy as np
import pandas as pd
from pathlib import Path
import math, os, sys
from rdkit import Chem
from scipy.interpolate import interp1d
import scipy.optimize as sco
from scipy import constants
from copy import copy


from .models import FH, NRTL, UNIQUAC, UNIFAC, QuarticModel
from .models.unifac import get_unifac_version
from .get_molecular_properties import load_molecular_properties, search_molecule
from .models.cem import CEM
from .kbi import mkdr
from .functions import get_solute_molid, mol2vol

def spinodal_fn(z, Hij):
	'''gets spinodal from roots of second derivative'''
	sign_changes = np.diff(np.sign(Hij))  # diff of signs between consecutive elements
	spin_inds = [s for s, sc in enumerate(sign_changes) if sc != 0 and ~np.isnan([sc]) if (s not in range(5)) and (s not in range(len(sign_changes)-5,len(sign_changes)))]
	if len(spin_inds) == 2:
		if z.shape[1] > 1:
			return z[spin_inds,:]
		else:
			return z[spin_inds]
	else:
		return None
	
def binodal_fn(num_comp, rec_steps, G_mix, activity_coefs, solute_ind, bi_min=-np.inf, bi_max=np.inf):
	''' get coexistence curve'''
	bi_obj = CEM(num_comp=num_comp, rec_steps=rec_steps, G_mix=G_mix, activity_coefs=activity_coefs)
	bi_vals = bi_obj.binodal_matrix_molfrac
	if num_comp < 3:
		bi_vals = bi_vals[:,:,solute_ind]
		if len(bi_vals) > 0:
			mask = np.all((bi_vals > bi_min) & (bi_vals < bi_max), axis=1)
			bi_filter = bi_vals[mask]
			try:
				bi1, bi2 = bi_filter.min(), bi_filter.max()
				return bi1, bi2
			except:
				return np.nan, np.nan
	else:
		return bi_vals
	return np.nan, np.nan

def numerical_binodal_fn(x, GM, dGM, sp1=None, sp2=None):
	'''	numerical binodal function that searches for binodal points that minimize a function for binary mixture '''	
	# if both spinodals are np.nan, don't execute binodal function
	if np.isnan(sp1) and np.isnan(sp2):
		return {}

	if np.all([sp1, sp2] == None):
		dGM = np.gradient(GM, x)
		d2GM = np.gradient(dGM, x)
		sps = spinodal_fn(x, d2GM)

	iiall = ~np.isnan(dGM) & ~np.isnan(GM)
	ii1 = (x < sp1) & ~np.isnan(dGM) & ~np.isnan(GM)
	ii2 = (x > sp2) & ~np.isnan(dGM) & ~np.isnan(GM)

	Gx_from_x = interp1d(x[iiall], GM[iiall], kind='cubic', fill_value="extrapolate")
	x2_from_dGx = interp1d(dGM[ii2], x[ii2])

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
	bis = sco.minimize(objective, x0=[0.1, 0.99], bounds=((1E-5,sp1),(sp2,0.9999)))

	bi1 = bis.x[0]
	bi2 = np.interp(dGx, dGM[ii2],  x[ii2])
	return bi1, bi2



def Tc_search(smiles: list, lle_type=None, Tmin=100, Tmax=500, dT=5, unif_version="unifac"):
	''' 
	find Tc for binary systems 

	if lle_type is specificied, calculation is more efficient
	'''
	if "md" in unif_version or 'kbi' in unif_version:
		version = "unifac-kbi"
	elif "il" in unif_version:
		version = "unifac-il"
	else:
		version = "unifac"

	i=0
	if lle_type is not None:
		T_initial = {"ucst": Tmax, "lcst": Tmin}
		T = T_initial[lle_type]
		dT = 50
		while T >= 0 and T <= 1200:
			model = UNIFAC(T=T, smiles=smiles, version=version)
			d2GM = model.det_Hij()
			sign_changes = np.diff(np.sign(d2GM))  # diff of signs between consecutive elements
			spin_inds = [s for s, sc in enumerate(sign_changes) if sc != 0 and ~np.isnan([sc])]
			if len(spin_inds) > 0 and dT == 50:
				if T == Tmax and i==0:
					T += 50
					i+=1
					continue
				if lle_type.lower() == "ucst":
					T += dT-20
				else:
					T -= dT-20
				dT = 20
			elif len(spin_inds) > 0 and dT == 20:
				if lle_type.lower() == "ucst":
					T += dT-5
				else:
					T -= dT-5
				dT = 5
			elif len(spin_inds) > 0 and dT == 5:
				return T
			else:
				if lle_type.lower() == "ucst":
					T -= dT
				else:
					T += dT
		return np.nan
	# if lle type is not known, calculate spinodals for all T
	# Tc is where there is the smallest difference between spinodal values
	else:
		T_values = np.arange(Tmin, Tmax+1, dT)
		sp_matrix = np.full((len(T_values), len(smiles)), fill_value=np.nan)
		for t, T in enumerate(T_values):
			model = UNIFAC(T=T, smiles=smiles, version=version)
			d2GM = model.det_Hij()
			sp = spinodal_fn(model.z, d2GM)
			if sp is not None:			
				sp_matrix[t] = [sp[:,0].min(), sp[:,0].max()]
			else:
				sp_matrix[t] = [np.nan, np.nan]
		# get Tc where there is the smallest difference between spinodal values
		try:
			nan_mask = ~np.isnan(sp_matrix[:,0]) & ~np.isnan(sp_matrix[:,1])
			sp_matrix_filter = sp_matrix[nan_mask]
			T_values_filter = T_values[nan_mask]
			crit_ind = np.abs(sp_matrix_filter[:,0]-sp_matrix_filter[:,1]).argmin()
			return T_values_filter[crit_ind]
		except:
			return np.nan

class ThermoModel:
	"""run thermo model on kbi results"""
	def __init__(
			self, 
			model_name: str, # thermo model
			KBIModel=None, # option to feed in kbi model
			Tmin=100, Tmax=400, dT=10, # temperature range
			quartic_gid_type = 'vol', # ideal gibbs type (x log(v) or x log(x))
		):

		self.kbi_model = KBIModel
		self.identifiers = self.kbi_model.unique_mols
		self.identifier_type = "mol_id"
		self.model_name = model_name.lower()
		self.save_dir = mkdr(f"{self.kbi_model.kbi_method_dir}/{self.model_name}/")

		# grab solute from KBIModel
		self.kbi_model.solute_mol = self.kbi_model.solute
		
		# get interaction parameters
		self.IP = None
		if self.model_name in ["nrtl", "uniquac"]:
			# get IP parameters for model type
			IP_map = {
				"nrtl": self.kbi_model.nrtl_taus, 
				"uniquac": self.kbi_model.uniquac_du, 
			}
			self.IP = IP_map[self.model_name]

		# initialize temperatures
		self.Tmin = Tmin
		self.Tmax = Tmax
		self.dT = dT

		# get type of thermo model
		model_map = {
			"fh": FH, 
			"nrtl": NRTL, 
			"uniquac": UNIQUAC, 
			"unifac": UNIFAC, 
			"quartic": QuarticModel, 
		}
		self.model_type = model_map[self.model_name]

		### initialize thermodynamic model
		# for unifac model
		if self.model_type == UNIFAC:
			self.unif_version = get_unifac_version('unifac', self.kbi_model.smiles)
			self.model = self.model_type(T=Tmax, smiles=self.kbi_model.smiles, version=self.unif_version)
		# uniquac model
		elif self.model_type == UNIQUAC:
			self.model = self.model_type(smiles=self.kbi_model.smiles, IP=self.IP)
		# quartic model
		elif self.model_type == QuarticModel:
			self.model = self.model_type(z_data=KBIModel.z, Hmix=KBIModel.Hmix, Sex=KBIModel.S_ex, molar_vol=self.kbi_model.molar_vol, gid_type=quartic_gid_type)
		# fh and nrtl model
		else: 
			self.model = self.model_type(smiles=self.kbi_model.smiles, IP=self.IP)

		# get volume fraction of composition
		self.z = self.model.z
		self.v = mol2vol(self.z, self.kbi_model.molar_vol)

		# for Flory-Huggins model, we need extra parameters for scaling mixing free energy with temperature
		if self.model_type == FH:
			self.model.load_thermo_data(phi=KBIModel.v[:,self.kbi_model.solute_loc], Smix=KBIModel.Smix, Hmix=KBIModel.Hmix)


	def run(self):
		# first perform temperature scaling to get: Gmix, spinodals, binodals
		if self.kbi_model.num_comp == 2:
			self._binary_temperature_scaling()
		else:
			self._multicomp_temperature_scaling()
		# then calculate saxs Io
		self._calculate_saxs_Io()

	@property
	def Rc(self):
		return constants.R / 1000

	@property
	def N_A(self):
		return constants.N_A

	@property
	def T_values(self):
		return np.arange(self.Tmin, self.Tmax+1E-3, self.dT)[::-1]

	def critical_point(self, sp_bi_vals, T):
		if self.model_type == FH:
			self.xc = self.model.xc
			self.phic = self.model.phic
		else:
			self.xc = np.mean(sp_bi_vals)
			self.phic = mol2vol(([self.xc, 1-self.xc]), self.kbi_model.molar_vol)
		self.Tc = T
	
	def _binary_temperature_scaling(self):

		self.GM = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		self.d2GM = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		self.x_sp = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
		self.x_bi = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
		self.GM_sp = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
		self.GM_bi = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
		self.v_sp = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
		self.v_bi = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
		
		for t, T in enumerate(self.T_values):
			# set variables to nan
			sp1, sp2, bi1, bi2 = np.empty(4)*np.nan

			# for just FH model
			if self.model_type == FH:
				self.GM[t,:] = self.model.GM(self.v[:,0], T)
				self.d2GM[t,:] = self.model.det_Hij(self.v[:,0],T)
				try:
					sp1, sp2, bi1, bi2 = self.model.ps_calc(self.model.phic, T)
				except:
					pass
			
			# for all other thermodynamic models
			else:
				# unifac, requires a new model object at each temperature
				if self.model_type == UNIFAC:
					self.model = self.model_type(T=T, smiles=self.kbi_model.smiles, version=self.unif_version)
					gm = self.model.GM()
					dgm = self.model.dGM_dxs().flatten()
					self.GM[t,:] = gm
					self.d2GM[t,:] = self.model.det_Hij()
					gammas = self.model.gammas()

				# other thermo models (uniquac, nrtl, quartic-numerical)
				else: 
					gm = self.model.GM(T)
					dgm = self.model.dGM_dxs(T)[:,self.kbi_model.solute_loc]
					self.GM[t,:] = gm
					self.d2GM[t,:] = self.model.det_Hij(T)
					gammas = self.model.gammas(T)

				# now get spinodals and binodals
				sp_vals = spinodal_fn(self.z, self.d2GM[t,:])
				if sp_vals is not None:		
					sp_vals = sp_vals[:,self.kbi_model.solute_loc]	
					sp1, sp2 = sp_vals.min(), sp_vals.max()
					# quartic doesn't have gammas --> uses the numerical binodal function
					if self.model_type in [NRTL]:
						bi1, bi2 = numerical_binodal_fn(x=self.z[:,self.kbi_model.solute_loc], sp1=sp1, sp2=sp2, GM=gm, dGM=dgm)
					# thermodynamic models can use convex envelope method, which requires Gmix and activity coefficients
					else:
						bi1, bi2 = binodal_fn(num_comp=self.kbi_model.num_comp, rec_steps=self.model.rec_steps, G_mix=gm, activity_coefs=gammas, solute_ind=self.kbi_model.solute_loc, bi_min=0.001, bi_max=0.99)

			self.x_sp[t,:] = [sp1, sp2]
			self.x_bi[t,:] = [bi1, bi2]

			if np.all(~np.isnan([sp1, sp2])):
				sp1_ind = np.abs(self.z[:,self.kbi_model.solute_loc] - sp1).argmin()
				sp2_ind = np.abs(self.z[:,self.kbi_model.solute_loc] - sp2).argmin()
				self.v_sp[t,:] = [self.v[sp1_ind, self.kbi_model.solute_loc], self.v[sp2_ind, self.kbi_model.solute_loc]]
				self.GM_sp[t,:] = [self.GM[t,sp1_ind], self.GM[t,sp2_ind]]
			
			if np.all(~np.isnan([bi1, bi2])):
				bi1_ind = np.abs(self.z[:,self.kbi_model.solute_loc] - bi1).argmin()
				bi2_ind = np.abs(self.z[:,self.kbi_model.solute_loc] - bi2).argmin()
				self.v_bi[t,:] = [self.v[bi1_ind, self.kbi_model.solute_loc], self.v[bi2_ind, self.kbi_model.solute_loc]]
				self.GM_bi[t,:] = [self.GM[t,bi1_ind], self.GM[t,bi2_ind]]
				
		# get critical point
		# find where there is the smallest difference between the spinodal values
		nan_mask = ~np.isnan(self.x_sp[:,0]) & ~np.isnan(self.x_sp[:,1])
		x_filter = self.x_sp[nan_mask]
		T_values_filter = self.T_values[nan_mask]
		crit_ind = np.abs(x_filter[:,0]-x_filter[:,1]).argmin()
		self.critical_point(x_filter[crit_ind,:], T_values_filter[crit_ind])
		# get lle type based on the spinodal values
		if T_values_filter[crit_ind] ==  T_values_filter.max():
			self.lle_type = "ucst"
		elif T_values_filter[crit_ind] == T_values_filter.min():
			self.lle_type = "lcst"

		# creates pd.DataFrame obj and save
		if self.save_dir is not None:
			df = pd.DataFrame()
			df["T"] = self.T_values
			for i in range(self.kbi_model.num_comp):
				df[f"x_sp{i+1}"] = self.x_sp[:,i]
				df[f"v_sp{i+1}"] = self.v_sp[:,i]
				df[f"x_bi{i+1}"] = self.x_bi[:,i]
				df[f"v_bi{i+1}"] = self.v_bi[:,i]
			df.to_csv(f"{self.save_dir}{self.model_name}_phase_instability_values.csv", index=False)


	def _multicomp_temperature_scaling(self):

		all_GMs = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		all_d2GMs = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		all_sp_ls = []
		all_bi_ls = []

		for t, T in enumerate(self.T_values):
			if self.model_name == "unifac":
				self.model = UNIFAC(T=T, smiles=self.kbi_model.smiles, version=self.unif_version)
				GM = self.model.GM()
				det_Hij = self.model.det_Hij()
				gammas = self.model.gammas()
			elif self.model_name in ["uniquac", "quartic"]:
				GM = self.model.GM(T)
				det_Hij = self.model.det_Hij(T)
				gammas = self.model.gammas(T)

			all_GMs[t] = GM
			if self.model_type != "unifac":
				all_d2GMs[t] = det_Hij
				# filter 2nd derivative for spinodal determination
				mask = (all_d2GMs[t] > -1) & (all_d2GMs[t] <= 1)
				sps = spinodal_fn(z=self.model.z, Hij=det_Hij)
				all_sp_ls += [sps]

			bi_vals = binodal_fn(num_comp=self.kbi_model.num_comp, rec_steps=self.model.rec_steps, G_mix=GM, activity_coefs=gammas, solute_ind=self.kbi_model.solute_loc, bi_min=0.001, bi_max=0.99)
			all_bi_ls += [bi_vals]
	
		self.d2GM = all_d2GMs
		self.GM = all_GMs
		self.x_sp = all_sp_ls
		self.x_bi = all_bi_ls


	def _calculate_saxs_Io(self):
		re2 = 7.9524E-26 # cm^2
		kb = self.Rc / self.N_A

		V_bar_i0 = self.z @ self.kbi_model.molar_vol
		N_bar = self.z @ self.kbi_model.n_electrons
		rho_e = 1/V_bar_i0

		drho_dx = np.zeros(self.z.shape[0])
		for j in range(self.kbi_model.num_comp-1):
			drho_dx += self.N_A*(rho_e*(self.kbi_model.n_electrons[j]-self.kbi_model.n_electrons[-1]) - N_bar*rho_e*(self.kbi_model.molar_vol[j]-self.kbi_model.molar_vol[-1])/V_bar_i0)

		I0_arr = np.empty((len(self.T_values), self.z.shape[0]))
		x_I0_max = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
		v_I0_max = np.full((len(self.T_values), self.kbi_model.num_comp), fill_value=np.nan)
		df_I0_max = pd.DataFrame(columns=['T', 'I0'])
		df_I0 = pd.DataFrame()
		df_I0['x'] = self.z[:,self.kbi_model.solute_loc]
		df_I0['v'] = self.v[:,self.kbi_model.solute_loc]

		for t, T in enumerate(self.T_values):
			S0 = kb * T * V_bar_i0 / self.d2GM[t]
			I0 = re2 * (drho_dx**2) * S0
			I0_arr[t] = I0
			I0 = np.array([np.round(i, 4) for i in I0])
			I0_filter = ~np.isnan(I0)
			if len(I0_filter) > 0:
				I0_max = max(I0[I0_filter])
				try:
					if self.lle_type == "ucst" and T > self.Tc:
						df_I0[T] = I0
						new_row = {"T": T, "I0": I0_max}
						df_I0_max = df_I0_max._append(new_row, ignore_index=True)
						v_I0_max[t] = self.v[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
						x_I0_max[t] = self.z[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
					elif self.lle_type == "lcst" and T < self.Tc:
						df_I0[T] = I0
						new_row = {"T": T, "I0": I0_max}
						df_I0_max = df_I0_max._append(new_row, ignore_index=True)
						v_I0_max[t] = self.v[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
						x_I0_max[t] = self.z[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
				except:
					pass
			
		self.I0_max = df_I0_max
		self.I0_arr = I0_arr

		# remove nan values, ie, for T <= Tc
		nan_filter = ~np.any(np.isnan(x_I0_max), axis=1)
		self.x_I0_max = x_I0_max[nan_filter]
		self.v_I0_max = v_I0_max[nan_filter]

		# convert to pandas df for saving
		if self.save_dir is not None:
			df_I0_max.to_csv(f"{self.save_dir}{self.model_name}_Io_max_values.csv", index=False)
			df_I0.to_csv(f"{self.save_dir}{self.model_name}_Io_values.csv", index=False)


class UNIFACThermoModel:
	"""run thermo model temperature scaling via unifac"""
	def __init__(
			self, 
			unif_version="unifac",
			solute_mol=None,
			identifiers=None, 
			identifier_type=None,
			Tmin=100, Tmax=400, dT=10,
			save_dir=None
		):

		self.save_dir = save_dir
		self.identifiers = identifiers
		self.identifier_type = identifier_type

		# initialize temperatures
		self.Tmin = Tmin
		self.Tmax = Tmax
		self.dT = dT

		if solute_mol is not None:
			self.solute_mol = solute_mol
		else:
			self.solute_mol = get_solute_molid(self.mol_id, self.mol_class)

		### initialize thermodynamic model
		# for unifac model
		self.unif_version = get_unifac_version(unif_version, self.smiles)
		self.model = UNIFAC(T=Tmax, smiles=self.smiles, version=self.unif_version)
		
		# get volume fraction of composition
		self.z = self.model.z
		self.v = mol2vol(self.z, self.molar_vol)


	def run(self):
		# first perform temperature scaling to get: Gmix, spinodals, binodals
		if self.num_comp == 2:
			self._binary_temperature_scaling()
			# then calculate saxs Io
			self._calculate_saxs_Io()
		else:
			self._multicomp_temperature_scaling()

	@property
	def Rc(self):
		return constants.R / 1000

	@property
	def N_A(self):
		return constants.N_A

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
	
	def critical_point(self, sp_bi_vals, T):
		self.xc = np.mean(sp_bi_vals)
		self.phic = mol2vol(([self.xc, 1-self.xc]), self.molar_vol)
		self.Tc = T
	
	@property
	def solute(self):
		return self.solute_mol

	@property
	def solute_loc(self):
		return list(self.mol_id).index(self.solute)
	
	@property
	def solute_name(self):
		return self.mol_name[self.solute_loc]
	
	def _binary_temperature_scaling(self):

		self.GM = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		self.d2GM = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		self.x_sp = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
		self.x_bi = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
		self.GM_sp = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
		self.GM_bi = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
		self.v_sp = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
		self.v_bi = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
		
		for t, T in enumerate(self.T_values):
			# set variables to nan
			sp1, sp2, bi1, bi2 = np.empty(4)*np.nan

			# unifac, requires a new model object at each temperature
			self.model = UNIFAC(T=T, smiles=self.smiles, version=self.unif_version)
			gm = self.model.GM()
			self.GM[t,:] = gm
			self.d2GM[t,:] = self.model.det_Hij()
			gammas = self.model.gammas()

			# now get spinodals and binodals
			sp_vals = spinodal_fn(self.z, self.d2GM[t,:])
			if sp_vals is not None:		
				sp_vals = sp_vals[:,self.solute_loc]	
				sp1, sp2 = sp_vals.min(), sp_vals.max()
				bi1, bi2 = binodal_fn(num_comp=self.num_comp, rec_steps=self.model.rec_steps, G_mix=gm, activity_coefs=gammas, solute_ind=self.solute_loc, bi_min=0.001, bi_max=0.99)

			self.x_sp[t,:] = [sp1, sp2]
			self.x_bi[t,:] = [bi1, bi2]

			if np.all(~np.isnan([sp1, sp2])):
				sp1_ind = np.abs(self.z[:,self.solute_loc] - sp1).argmin()
				sp2_ind = np.abs(self.z[:,self.solute_loc] - sp2).argmin()
				self.v_sp[t,:] = [self.v[sp1_ind, self.solute_loc], self.v[sp2_ind, self.solute_loc]]
				self.GM_sp[t,:] = [self.GM[t,sp1_ind], self.GM[t,sp2_ind]]
			
			if np.all(~np.isnan([bi1, bi2])):
				bi1_ind = np.abs(self.z[:,self.solute_loc] - bi1).argmin()
				bi2_ind = np.abs(self.z[:,self.solute_loc] - bi2).argmin()
				self.v_bi[t,:] = [self.v[bi1_ind, self.solute_loc], self.v[bi2_ind, self.solute_loc]]
				self.GM_bi[t,:] = [self.GM[t,bi1_ind], self.GM[t,bi2_ind]]
				
		# get critical point
		# find where there is the smallest difference between the spinodal values
		nan_mask = ~np.isnan(self.x_sp[:,0]) & ~np.isnan(self.x_sp[:,1])
		x_filter = self.x_sp[nan_mask]
		T_values_filter = self.T_values[nan_mask]
		crit_ind = np.abs(x_filter[:,0]-x_filter[:,1]).argmin()
		self.critical_point(x_filter[crit_ind,:], T_values_filter[crit_ind])
		# get lle type based on the spinodal values
		if T_values_filter[crit_ind] ==  T_values_filter.max():
			self.lle_type = "ucst"
		elif T_values_filter[crit_ind] == T_values_filter.min():
			self.lle_type = "lcst"

	def _multicomp_temperature_scaling(self):

		all_GMs = np.full((len(self.T_values), self.z.shape[0]), fill_value=np.nan)
		all_bi_ls = []

		for t, T in enumerate(self.T_values):
			self.model = UNIFAC(T=T, smiles=self.smiles, version=self.unif_version)
			GM = self.model.GM()
			gammas = self.model.gammas()
			all_GMs[t] = GM

			bi_vals = binodal_fn(num_comp=self.num_comp, rec_steps=self.model.rec_steps, G_mix=GM, activity_coefs=gammas, solute_ind=self.solute_loc, bi_min=0.001, bi_max=0.99)
			all_bi_ls += [bi_vals]
	
		self.GM = all_GMs
		self.x_bi = all_bi_ls

	def _calculate_saxs_Io(self):
		re2 = 7.9524E-26 # cm^2
		kb = self.Rc / self.N_A

		V_bar_i0 = self.z @ self.molar_vol
		N_bar = self.z @ self.n_electrons
		rho_e = 1/V_bar_i0

		drho_dx = np.zeros(self.z.shape[0])
		for j in range(self.num_comp-1):
			drho_dx += self.N_A*(rho_e*(self.n_electrons[j]-self.n_electrons[-1]) - N_bar*rho_e*(self.molar_vol[j]-self.molar_vol[-1])/V_bar_i0)

		I0_arr = np.empty((len(self.T_values), self.z.shape[0]))
		x_I0_max = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
		v_I0_max = np.full((len(self.T_values), self.num_comp), fill_value=np.nan)
		df_I0_max = pd.DataFrame(columns=['T', 'I0'])
		df_I0 = pd.DataFrame()
		df_I0['x'] = self.z[:,self.solute_loc]
		df_I0['v'] = self.v[:,self.solute_loc]

		for t, T in enumerate(self.T_values):
			S0 = kb * T * V_bar_i0 / self.d2GM[t]
			I0 = re2 * (drho_dx**2) * S0
			I0_arr[t] = I0
			I0 = np.array([np.round(i, 4) for i in I0])
			I0_filter = ~np.isnan(I0)
			if len(I0_filter) > 0:
				I0_max = max(I0[I0_filter])
				try:
					if self.lle_type == "ucst" and T > self.Tc:
						df_I0[T] = I0
						new_row = {"T": T, "I0": I0_max}
						df_I0_max = df_I0_max._append(new_row, ignore_index=True)
						v_I0_max[t] = self.v[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
						x_I0_max[t] = self.z[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
					elif self.lle_type == "lcst" and T < self.Tc:
						df_I0[T] = I0
						new_row = {"T": T, "I0": I0_max}
						df_I0_max = df_I0_max._append(new_row, ignore_index=True)
						v_I0_max[t] = self.v[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
						x_I0_max[t] = self.z[I0_filter][np.where(I0[I0_filter]==I0_max)[0][0]]
				except:
					pass
			
		self.I0_max = df_I0_max
		self.I0_arr = I0_arr

		# remove nan values, ie, for T <= Tc
		nan_filter = ~np.any(np.isnan(x_I0_max), axis=1)
		self.x_I0_max = x_I0_max[nan_filter]
		self.v_I0_max = v_I0_max[nan_filter]
