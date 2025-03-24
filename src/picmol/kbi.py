import numpy as np
import pandas as pd
import glob, copy
from scipy.integrate import trapz, quad
from scipy.optimize import curve_fit
import os, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
plt.style.use(Path(__file__).parent / 'presentation.mplstyle')

from scipy.constants import R, pi

from .get_molecular_properties import load_molecular_properties
from .models.uniquac import UNIQUAC_RQ, fit_du_to_Hmix
from .models import UNIQUAC, UNIFAC, QuarticModel, FH
from .models.cem import PointDisc
from .conversions import mol2vol


def mkdr(dir_path):
	# creates a new directory and assigns it a variable
	if os.path.exists(dir_path) == False:
		os.mkdir(dir_path)
	return dir_path

def add_zeros(arr):
	# adds zeros before 1st and last position --> for Gmix, etc. 
	f = np.zeros(len(arr) + 2)
	f[1:-1] = arr 
	return f

class KBI:

	def __init__(
			self, 
			prj_path: str, 
			pure_component_path: str,
			rdf_dir: str = "rdf_files", 
			kbi_method: str = "adj", 
			kbi_fig_dirname: str = "kbi_analysis",
			avg_start_time = 100, 
			avg_end_time = None,
		):

		# assumes folder organization is: project / systems / rdfs
		self.prj_path = prj_path

		# location of pure component files
		self.pure_component_dir = pure_component_path

		self.avg_start_time = round(1000 * avg_start_time) # start time in [ps] for enthalpy, volume, density averaging
		if avg_end_time is not None:
			self.avg_end_time = round(1000 * avg_end_time) # end time in [ps] for enthalpy, volume, density averaging
		else:
			self.avg_end_time = None
		
		# get kbi method: raw, adj, kgv
		self.kbi_method = kbi_method.lower()
		self.kbi_fig_dirname = kbi_fig_dirname

		# setup other folders
		self.setup_kbi_folders()

		# rdfs need to be located in their corresponding system folder in a subdirectory
		# assumes that there is only 1 rdf file with "mol1" and "mol2" in filename
		# assumes that rdf files are stored in a text file type (i.e., can be loaded with np.loadtxt) with x=r and y=g
		self.rdf_dir = rdf_dir

		# for folder to be considered a system, it needs to have a .top file
		self.systems = [sys for sys in os.listdir(self.prj_path) if os.path.isdir(os.path.join(self.prj_path,sys)) and f"{sys}.top" in os.listdir(f"{self.prj_path}/{sys}/")]
		# sort systems so in order by solute moleclule number
		self.sort_systems()

		# get number of systems in project
		self.n_sys = len(self.systems)

		# get gas constant in kJ/mol-K
		self.Rc = R / 1000 # kJ / mol K


	def get_time_average(self, time, arr):
		'''get time average from property using .edr files'''
		start_ind = np.abs(time - self.avg_start_time).argmin()
		if self.avg_end_time is not None:
			end_ind = np.abs(time - self.avg_end_time).argmin()
			return np.mean(arr[start_ind:end_ind])
		else:
			return np.mean(arr[start_ind:])


	def get_simulation_temps(self):
		'''get actual simulation temperature for each system'''
		sys_Tsims = np.zeros(self.n_sys)
		for s, sys in enumerate(self.systems):
			os.chdir(f"{self.prj_path}/{sys}/")
			# get .edr file
			npt_edr_files = [file for file in os.listdir(f'.') if (sys in file) and ("npt" in file) and ("edr" in file)]
			if len(npt_edr_files) > 1:
				npt_edr_file = f"{sys}_npt.edr"
			else:
				npt_edr_file = npt_edr_files[0]
			# get temperature from .edr file
			if os.path.exists('temperature.xvg') == False:
				os.system(f"echo temperature | gmx energy -f {npt_edr_file} -o temperature.xvg")
			# average temperatures over time
			time, T = np.loadtxt('temperature.xvg', comments=["#", "@"], unpack=True)
			sys_Tsims[s] = self.get_time_average(time, T)
			
		# write temperature to .txt
		with open(f'{self.kbi_dir}Tsim.txt', 'w') as f:
			f.write('system,T_actual\n')
			for t, T in enumerate(sys_Tsims):
				f.write(f'{self.systems[t]},{T:.4f}\n')
		self._Tsim_all_systems = sys_Tsims
		return self._Tsim_all_systems

	@property
	def T_sim(self):
		try:
			self._Tsim_all_systems
		except AttributeError:
			self.get_simulation_temps()
		return round(np.mean(self._Tsim_all_systems))

	def setup_kbi_folders(self):
		'''create folders for kbi analysis'''
		mkdr(f"{self.prj_path}/figures/")
		self.kbi_dir = mkdr(f"{self.prj_path}/figures/{self.kbi_fig_dirname}/")
		self.kbi_method_dir = mkdr(f"{self.kbi_dir}/{self.kbi_method}_kbi_method/")
		self.kbi_indiv_fig_dir = mkdr(f"{self.kbi_method_dir}/indiv_kbi/")
		self.kbi_indiv_data_dir = mkdr(f"{self.kbi_method_dir}/kbi_data/")

	def read_top(self, sys_parent_dir, sys):
		'''extract molecules and number of molecules in top file'''
		sys_mols = []
		sys_total_num_mols = 0
		sys_mol_nums_by_component = {}
		with open(f"{sys_parent_dir}/{sys}/{sys}.top", "r") as top:
			lines = top.readlines()
			for l, line in enumerate(lines):
				# get line that contains "molecules"
				if "molecule" in line:
					molecules_lines = lines[l+1:]
			top.close()
		for line in molecules_lines:
			if len(line.split()) > 0:
				# check that first split contains alpha characters
				alpha_chk = [char.isalpha() for char in line.split()[0]]
				# if alpha characters are found, append molecules
				if True in alpha_chk:
					mol = line.split()[0]
					sys_mols.append(mol)
					sys_mol_nums_by_component[mol] = int(line.split()[1])
					sys_total_num_mols += int(line.split()[1])
		return sys_mols, sys_total_num_mols, sys_mol_nums_by_component
	

	def extract_sys_info_from_top(self):
		'''create dictionary containing top info for each system in project'''
		mols_present = []
		total_num_mols = {sys: 0 for sys in self.systems}
		mol_nums_by_component = {sys: {} for sys in self.systems}
		for sys in self.systems:
			sys_mols, sys_total_num_mols, sys_mol_nums_by_component = self.read_top(sys_parent_dir=self.prj_path, sys=sys)
			mols_present.append(sys_mols)
			total_num_mols[sys] = sys_total_num_mols
			mol_nums_by_component[sys] = sys_mol_nums_by_component
		# get unique mols in the system
		unique_mols = np.unique(mols_present)
		self.top_info = {
			"unique_mols": unique_mols, 
			"mol_nums_by_component": mol_nums_by_component, 
			"total_num_mols": total_num_mols
		}

	@property
	def unique_mols(self):
		'''unique mol_ids in project'''
		try:
			self.top_info
		except AttributeError:
			self.extract_sys_info_from_top()
		return self.top_info["unique_mols"]
		
	@property
	def mol_nums_by_component(self):
		'''molecule numbers of individual components'''
		try:
			self.top_info
		except AttributeError:
			self.extract_sys_info_from_top()
		return self.top_info["mol_nums_by_component"]
	
	@property
	def total_num_mols(self):
		'''total number of molecules in each system'''
		try:
			self.top_info
		except AttributeError:
			self.extract_sys_info_from_top()
		return self.top_info["total_num_mols"]
	
	@property
	def properties_by_molid(self):
		'''read in molecular_properties file and set mol_id as index'''
		prop_df = load_molecular_properties("mol_id")
		return prop_df.loc[self.unique_mols, :]
	
	@property
	def mol_name_dict(self):
		return {mol: self.properties_by_molid["mol_name"][mol] for mol in self.unique_mols}
		
	@property
	def mol_class_dict(self):
		return {mol: self.properties_by_molid["mol_class"][mol] for mol in self.unique_mols}

	@property
	def mol_smiles_dict(self):
		return {mol: self.properties_by_molid["smiles"][mol] for mol in self.unique_mols}
	
	@property
	def molar_vol(self):
		return self.properties_by_molid["molar_vol"].to_numpy()

	@property
	def mw(self):
		return self.properties_by_molid["mol_wt"].to_numpy()
	
	@property
	def smiles(self):
		return self.properties_by_molid["smiles"].values

	def assign_solute_mol(self):
		'''finds the molecule that should be considered as solute; priority goes to: 1. extracant, 2. modifier, 3. solute, 4. solvent'''
		solute_mol = None
		# first check for extractant
		for mol in self.unique_mols:
			if self.mol_class_dict[mol] == 'extractant':
				solute_mol = mol
		# then check for modifier
		if solute_mol is None:
			for mol in self.unique_mols:
				if self.mol_class_dict[mol] == 'modifier':
					solute_mol = mol
		# then check for solute
		if solute_mol is None:
			for mol in self.unique_mols:
				if self.mol_class_dict[mol] == 'solute':
					solute_mol = mol
		# then solvent
		if solute_mol is None:
			for mol in self.unique_mols:
				if self.mol_class_dict[mol] == 'solvent':
					solute_mol = mol

		self.solute_mol = solute_mol
		return self.solute_mol
	
	@property
	def solute(self):
		'''get solute mol_id'''
		try:
			self.solute_mol
		except AttributeError: 
			self.assign_solute_mol()
		return self.solute_mol
	
	@property
	def solute_loc(self):
		'''get index of solute molecule'''
		return np.where(self.unique_mols==self.solute)[0][0]
	
	@property
	def solute_name(self):
		'''get name of solute molecule'''
		return self.mol_name_dict[self.solute]
	
	def sort_systems(self):
		'''sort systems based on molar fraction'''
		sys_df = pd.DataFrame({
			"systems": self.systems,
			"mols": [self.mol_nums_by_component[sys][self.solute] for sys in self.systems]
		})
		sys_df = sys_df.sort_values("mols").reset_index(drop=True)
		self.systems = sys_df["systems"].to_list()


	def system_compositions(self):
		"""get system properties at each composition"""
		df_comp = pd.DataFrame()

		for s, sys in enumerate(self.systems):
			# change to system directory
			os.chdir(f"{self.prj_path}/{sys}/")
			# get mols present in system
			mols_present = list(self.mol_nums_by_component[sys].keys())

			# get npt edr files for system properties; volume, enthalpy.
			npt_edr_files = [file for file in os.listdir('.') if (sys in file) and ("npt" in file) and ("edr" in file)]
			if len(npt_edr_files) > 1:
				npt_edr_file = f"{sys}_npt.edr"
			else:
				npt_edr_file = npt_edr_files[0]
			
			# get box volume in simulation
			if os.path.exists('volume.xvg') == False:
				os.system(f"echo volume | gmx energy -f {npt_edr_file} -o volume.xvg")
			time, V = np.loadtxt('volume.xvg', comments=["#", "@"], unpack=True)
			box_vol_nm3 = self.get_time_average(time, V)

			# get simulation enthalpy
			if os.path.exists('enthalpy_npt.xvg') == False:
				os.system(f"echo enthalpy | gmx energy -f {npt_edr_file} -o enthalpy_npt.xvg")
			time, H = np.loadtxt('enthalpy_npt.xvg', comments=["#", "@"], unpack=True)
			Hsim_kJ = self.get_time_average(time, H)/self.total_num_mols[sys]

			# add properties to dataframe
			df_comp.loc[s, "systems"] = sys
			for m, mol in enumerate(mols_present):
				df_comp.loc[s, f"x_{mol}"] = self.mol_nums_by_component[sys][mol] / self.total_num_mols[sys]
				df_comp.loc[s, f"phi_{mol}"] = (self.mol_nums_by_component[sys][mol] * self.molar_vol[m]) / sum(np.array(list(self.mol_nums_by_component[sys].values())) * self.molar_vol)
				df_comp.loc[s, f"c_{mol}_M"] = (self.mol_nums_by_component[sys][mol] / box_vol_nm3) * (10**24) / (6.022E+23)
				df_comp.loc[s, f'rho_{mol}'] = self.mol_nums_by_component[sys][mol] / box_vol_nm3
				df_comp.loc[s, f'n_{mol}'] = self.mol_nums_by_component[sys][mol]
			df_comp.loc[s, 'n_tot'] = self.total_num_mols[sys]
			df_comp.loc[s, 'box_vol'] = box_vol_nm3
			df_comp.loc[s, 'enthalpy'] = Hsim_kJ
			# replace all NaN values with zeros
			df_comp.fillna(0, inplace=True)

			# save to csv
			df_comp.to_csv(f'{self.kbi_dir}system_compositions.csv', index=False)

			self.df_comp = df_comp

	def kbi_fn(self, r, g, r_lo, r_hi, avg, r_max, sys_num, mol_1, mol_2, method):
		'''calculate the KBI depending on the method'''
		# filter r and g(r)
		r_filter = (r >= r_lo) & (r <= r_hi)
		r_filt = r[r_filter]
		g_filt = g[r_filter]

		# get molecular properties
		rho_mol1 = float(self.df_comp.loc[sys_num,f'rho_{mol_1}'])
		rho_mol2 = float(self.df_comp.loc[sys_num,f'rho_{mol_2}'])
		c_mol1 = float(self.df_comp.loc[sys_num, f'c_{mol_1}_M'])
		N_mol2 = int(self.df_comp.loc[sys_num, f'n_{mol_2}'])
		box_vol = float(self.df_comp.loc[sys_num, f'box_vol'])
		kd = int(mol_1 == mol_2)

		# get spacing between datapoints
		dr = 0.002 # GROMACS default
		# adjustment: ie., correct for g(r) != 1 at R
		if method == 'adj':
			h = g_filt - avg
		# no correction
		elif method == 'raw':
			h = g_filt - 1
		# apply damping function
		elif 'kgv' in method:
			# number of solvent molecules
			Nj = N_mol2
			# 1-volume ratio
			vr = 1 - ((4/3)*pi*r_filt**3/box_vol) 
			# coordination number of mol_2 surrounding mol_1
			cn = 4 * pi * r_filt**2 * rho_mol2 * (g_filt - 1)
			dNij = trapz(cn, x=r_filt, dx=dr)	
			# g(r) correction using Ganguly - van der Vegt approach
			g_gv_correct = g_filt * Nj * vr / (Nj * vr - dNij - kd) 
			# combo of g(r) correction with damping function K. 
			damp_k = (1 - (3*r_filt)/(2*r_max) + r_filt**3/(2*r_max**3))
			h = damp_k * (g_gv_correct - 1)
		
		f = 4 * pi * r_filt**2 * h
		kbi_nm3 = trapz(f, x=r_filt, dx=dr)
		kbi_cm3_mol = trapz(f, x=r_filt, dx=dr) * rho_mol1 * 1000 / c_mol1

		return kbi_nm3, kbi_cm3_mol

	def kbi_analysis(self):
		'''perform kbi analysis'''
		# get system compositions
		try:
			self.df_comp
		except AttributeError: 
			self.system_compositions()
		
		# create dataframes for each pairwise interaction
		df_kbi = pd.DataFrame()
		df_kbi[f"x_{self.solute}"] = self.df_comp[f"x_{self.solute}"]
		df_kbi[f"phi_{self.solute}"] = self.df_comp[f"phi_{self.solute}"]
		for i, mol_1 in enumerate(self.unique_mols):
			for j, mol_2 in enumerate(self.unique_mols):
				if i <= j:
					df_kbi[f'G_{mol_1}_{mol_2}_nm3'] = np.zeros(self.n_sys)
					df_kbi[f'G_{mol_1}_{mol_2}_cm3_mol'] = np.zeros(self.n_sys)

		for s, sys in enumerate(self.systems):
			# create kbi dataframe for each system for storing kbi's as a function of r
			df_kbi_sys = pd.DataFrame()
			# iterate through all possible molecular combinations with no repeats
			for i, mol_1 in enumerate(list(self.mol_nums_by_component[sys].keys())):
				for j, mol_2 in enumerate(list(self.mol_nums_by_component[sys].keys())):
					if i <= j:
						# read rdf file
						rdf_file = glob.glob(f"{self.prj_path}/{sys}/{self.rdf_dir}/*{mol_1}*{mol_2}*")[0]
						r, g = np.loadtxt(rdf_file, comments=["@", "#"], unpack=True)
						r = r[:-3]
						g = g[:-3]
						
						# get r_max and r_avg for kbi input
						r_avg = r[-1] - 0.5 
						# get limit g(r) for r --> R
						limit_g_not_1 = np.mean(g[r > r_avg])
						if np.isnan(limit_g_not_1):
							r2avg = np.round(r[-1]-1, 3)
							limit_g_not_1 = g[np.where(r == r2avg)[0][0]]
						
						# get kbis as a function of r
						kbi_cm3_mol_r = np.full((len(r)), fill_value=np.nan)
						kbi_nm3_r = np.full((len(r)), fill_value=np.nan)
						kbi_cm3_mol_sum = 0.
						kbi_nm3_sum = 0.
						for k in range(len(r)-1):
							kbi_nm3, kbi_cm3_mol = self.kbi_fn(r=r, g=g, r_lo=r[k], r_hi=r[k+1], r_max=max(r), sys_num=s, avg=limit_g_not_1, mol_1=mol_1, mol_2=mol_2, method=self.kbi_method)
							kbi_cm3_mol_sum += kbi_cm3_mol
							kbi_nm3_sum += kbi_nm3
							kbi_cm3_mol_r[k] = kbi_cm3_mol_sum
							kbi_nm3_r[k] = kbi_nm3_sum
						
						# add kbi values to nested dictionaries
						df_kbi.loc[s, f'G_{mol_1}_{mol_2}_nm3'] = kbi_nm3_sum
						df_kbi.loc[s, f'G_{mol_1}_{mol_2}_cm3_mol'] = kbi_cm3_mol_sum

						# save kbi's as a function of r as csv.
						df_kbi_sys['r'] = r 
						df_kbi_sys[f'G_{mol_1}_{mol_2}_nm3'] = kbi_nm3_r
						df_kbi_sys[f'G_{mol_1}_{mol_2}_cm3_mol'] = kbi_cm3_mol_r
						df_kbi_sys.to_csv(f'{self.kbi_indiv_data_dir}{sys}_kbis.csv', index=False)
						# create attribute for kbis as a function of r for each system
						setattr(self, f"kbi_{s}", df_kbi_sys)

		df_kbi.to_csv(f"{self.kbi_method_dir}kbis.csv", index=False)
		self.df_kbi = df_kbi

	@property
	def ij_combo(self):
		# get max number of rdfs per system
		sys_combo = []
		for sys in self.systems:
			sys_combo.append(sum([1 for file in os.listdir(f"{self.prj_path}/{sys}/{self.rdf_dir}/")]))
		return max(sys_combo)
	
	@property
	def z(self):
		try:
			self.df_comp
		except AttributeError: 
			self.system_compositions()
		z_mat = np.empty((self.df_comp.shape[0], len(self.unique_mols)))
		for i, mol in enumerate(self.unique_mols):
			z_mat[:,i] = self.df_comp[f'x_{mol}'].to_numpy()
		return z_mat

	@property
	def v(self):
		return mol2vol(self.z, self.molar_vol)
	
	@property
	def v0(self):
		v0 = np.zeros((self.v.shape[0]+2, self.v.shape[1]))
		v0[0,:] = [0,1]
		v0[1:-1,:] = self.v
		v0[-1,:] = [1,0]
		return v0
	
	@property
	def Hsim(self):
		try:
			self.df_comp
		except AttributeError: 
			self.system_compositions()
		return self.df_comp["enthalpy"].to_numpy()

	@property
	def G_matrix(self):
		''' create a symmetric matrix from KBI values '''
		try:
			self.df_kbi
		except:
			self.kbi_analysis()
		try:
			self.df_comp
		except AttributeError: 
			self.system_compositions()
		G = np.full((self.z.shape[0], len(self.unique_mols), len(self.unique_mols)), fill_value=np.nan)
		for i, mol_1 in enumerate(self.unique_mols):
			for j, mol_2 in enumerate(self.unique_mols):
				if i <= j:
					# fill matrix with kbi values in nm^3
					G[:,i,j] = 	self.df_kbi[f'G_{mol_1}_{mol_2}_nm3'].to_numpy()
					# the matrix should be symmetrical
					if i != j:
						G[:,j,i] = G[:,i,j]
		return G

	@property
	def B_matrix(self):
		B = np.full((self.z.shape[0],len(self.unique_mols), len(self.unique_mols)), fill_value=np.nan)
		for i, mol_1 in enumerate(self.unique_mols):
			rho_i = self.df_comp[f'rho_{mol_1}'].to_numpy()
			for j, mol_2 in enumerate(self.unique_mols):
				rho_j = self.df_comp[f'rho_{mol_2}'].to_numpy()
				kd_ij = int(i==j)
				B[:,i,j] = rho_i * rho_j * self.G_matrix[:,i,j] + rho_i * kd_ij
		return B

	@property 
	def det_B_matrix(self):
		'''get determinant of matrix B'''
		return np.linalg.det(self.B_matrix)
	
	@property
	def cofactors_Bij(self):
		'''get the cofactors of matrix B'''
		B_ij = np.full((self.z.shape[0], len(self.unique_mols), len(self.unique_mols), len(self.unique_mols)-1, len(self.unique_mols)-1), fill_value=np.nan)
		for i in range(len(self.unique_mols)):
			for j in range(len(self.unique_mols)):
				BB = (-1)**(i+j) * np.delete(np.delete(self.B_matrix, i, axis=1), j, axis=2)
				B_ij[:,i,j] = BB.reshape(B_ij[:,i,j].shape)
		return B_ij
	
	@property
	def dmu_dxs(self):
		'''chemical potential derivatives'''
		b_lower = np.zeros(self.z.shape[0])
		for i, mol_1 in enumerate(self.unique_mols):
			rho_i = self.df_comp[f'rho_{mol_1}'].to_numpy()
			for j, mol_2 in enumerate(self.unique_mols):
				rho_j = self.df_comp[f'rho_{mol_2}'].to_numpy()
				b_lower += rho_i * rho_j * np.linalg.det(self.cofactors_Bij[:,i,j])
		
		# get system properties
		V = self.df_comp["box_vol"].to_numpy()
		n_tot = self.df_comp["n_tot"].to_numpy()

		# chemical potential derivative wrt molecule number
		dmu_dN = np.full((self.z.shape[0], len(self.unique_mols), len(self.unique_mols)), fill_value=np.nan)
		for a in range(len(self.unique_mols)):
			for b in range(len(self.unique_mols)):
				b_upper = np.zeros(self.z.shape[0])
				for i, mol_1 in enumerate(self.unique_mols):
					rho_i = self.df_comp[f'rho_{mol_1}'].to_numpy()
					for j, mol_2 in enumerate(self.unique_mols):
						rho_j = self.df_comp[f'rho_{mol_2}'].to_numpy()
						b_upper += rho_i * rho_j * np.linalg.det((self.cofactors_Bij[:,a,b]*self.cofactors_Bij[:,i,j] - self.cofactors_Bij[:,i,a]*self.cofactors_Bij[:,j,b]))
				b_frac = b_upper/b_lower
				dmu_dN[:,a,b] = b_frac/(V*self.det_B_matrix)
		
		# convert to mol fraction
		dmu_dxs = np.full((self.z.shape[0],len(self.unique_mols)-1, len(self.unique_mols)-1), fill_value=np.nan)
		for i in range(len(self.unique_mols)-1):
			for j in range(len(self.unique_mols)-1):
				dmu_dxs[:,i,j] = n_tot * (dmu_dN[:,i,j] - dmu_dN[:,i,-1])
		
		# now get the derivative for each component
		dmui_dxi = np.full((self.z.shape[0], len(self.unique_mols)), fill_value=np.nan)
		dmui_dxi[:,:-1] = np.diagonal(dmu_dxs, axis1=1, axis2=2)
		sum_xi_dmui = np.zeros(self.z.shape[0])
		for i in range(len(self.unique_mols)-1):
			sum_xi_dmui += self.z[:,i] * dmui_dxi[:,i]
		dmui_dxi[:,-1] = sum_xi_dmui / self.z[:,-1]

		return dmui_dxi
		
	@property 
	def dlngamma_dxs(self):
		return self.dmu_dxs - 1/self.z
	
	def _get_ref_state(self, mol):
		# get mol index
		i = np.where(self.unique_mols == mol)[0][0] 
		# get max mol fr at each composition
		comp_max = self.z.max(axis=1) 
		# get mask for max mol frac at each composition
		is_max = self.z[:,i] == comp_max 
		# check if the mol is largest mol frac at any composition
		# if mol is max at any composition -- it can't be a solute
		if np.any(is_max):
			return "pure_component" 
		# if solute use infinite dilution reference state
		else:
			return "inf_dilution"

	@property
	def gammas(self):
		''' numerical integration of activity coefs.'''
		dlny = self.dlngamma_dxs
		int_dlny_dx = np.zeros(self.z.shape)

		for i, mol in enumerate(self.unique_mols):
			x = self.z[:, i]
			dlnyi = dlny[:, i]

			# determine if ref state is pure component
			ref_state = self._get_ref_state(mol)
			if ref_state == "pure_component":
				initial_x = [1, 0]
				sort_idx = -1
			else:
				initial_x = [0, 0]
				sort_idx = 1

			# set up array
			int_arr = np.zeros((self.z.shape[0] + 1, 3))
			int_arr[:-1, 0] = x
			int_arr[:-1, 1] = dlnyi
			int_arr[-1, :2] = initial_x
			# sort based on mol frac
			sorted_inds = np.argsort(int_arr[:, 0])[::sort_idx]
			int_arr = int_arr[sorted_inds]

			# numerical integration
			y0 = 0
			for j in range(1, self.z.shape[0] + 1):
				if j > 1:
					y0 = int_arr[j - 1, 2]
				# uses midpoint rule, ie., trapezoid method for numerical integration
				int_arr[j, 2] = y0 + 0.5 * (int_arr[j, 1] + int_arr[j - 1, 1]) * (int_arr[j, 0] - int_arr[j - 1, 0])

			# delete pure component point
			x0_idx = np.where(int_arr[:, 0] == initial_x[0])[0][0]
			int_arr = np.delete(int_arr, x0_idx, axis=0)

			# Ensure gammas index matches initial x index
			if not np.array_equal(x, int_arr[:, 0]):  # Check if x and integrated x match
				# Create a dictionary to map x values to integrated values
				x_to_integrated = {val: integrated for val, integrated in zip(int_arr[:, 0], int_arr[:, 2])}
				# Reconstruct the integrated array in the original order of x
				reordered_integrated = np.array([x_to_integrated[val] for val in x])
				# get exponential and add to array
				int_dlny_dx[:, i] = np.exp(reordered_integrated)
			else:
				int_dlny_dx[:, i] = np.exp(int_arr[:, 2])

		return int_dlny_dx

	@property
	def G_ex(self):
		return self.Rc * self.T_sim * (np.log(self.gammas) * self.z).sum(axis=1)
	
	def G_id(self, x1_mat, x2_mat):
		return self.Rc * self.T_sim * (x1_mat * np.log(x2_mat)).sum(axis=1)
	
	@property
	def G_mix_xv(self):
		return self.G_ex + self.G_id(self.z, self.v)

	@property
	def G_mix_xx(self):
		return self.G_ex + self.G_id(self.z, self.z)
	
	@property
	def G_mix_vv(self):
		return self.G_ex + self.G_id(self.v, self.v)

	@property
	def Hsim_pc(self):
		# first calculate H_pc for each component in system
		H_pc = {mol: 0 for mol in self.unique_mols}
		for i, mol in enumerate(self.unique_mols):
			# try and find directory
			sys = f"{mol}_{self.T_sim}"
			os.chdir(f"{self.pure_component_dir}/{sys}/")
			try:
				mols_present, total_num_mols, mol_nums_by_component = self.read_top(sys_parent_dir=self.pure_component_dir, sys=sys)
				# get npt edr files for system properties; volume, enthalpy.
				npt_edr_files = [file for file in os.listdir('.') if f"{sys}_npt" and "edr" in file]
				if len(npt_edr_files) > 1:
					npt_edr_file = f"{sys}_npt.edr"
				else:
					npt_edr_file = npt_edr_files[0]
				# get simulation enthalpy
				if os.path.exists('enthalpy_npt.xvg') == False:
					os.system(f"echo enthalpy | gmx energy -f {npt_edr_file} -o enthalpy_npt.xvg")
				time, H = np.loadtxt('enthalpy_npt.xvg', comments=["#", "@"], unpack=True)
				H_pc[mol] = self.get_time_average(time, H)/total_num_mols		
			except:
				# if file/path does not exist just use nan values
				H_pc[mol] = np.nan
		return H_pc

	@property
	def sim_molar_vol(self):
		# get molar volume of pure components
		vol = np.zeros(self.unique_mols.size)
		for i, mol in enumerate(self.unique_mols):
			# try and find directory
			sys = f"{mol}_{self.T_sim}"
			os.chdir(f"{self.pure_component_dir}/{sys}/")
			mols_present, total_num_mols, mol_nums_by_component = self.read_top(sys_parent_dir=self.pure_component_dir, sys=sys)
			# get npt edr files for system properties; volume, enthalpy.
			npt_edr_files = [file for file in os.listdir('.') if f"{sys}_npt" and "edr" in file]
			if len(npt_edr_files) > 1:
				npt_edr_file = f"{sys}_npt.edr"
			else:
				npt_edr_file = npt_edr_files[0]
			# get simulation density
			if os.path.exists('density_npt.xvg') == False:
				os.system(f"echo density | gmx energy -f {npt_edr_file} -o density_npt.xvg")
			time, rho = np.loadtxt('density_npt.xvg', comments=["#","@"], unpack=True)
			density = self.get_time_average(time, rho) / 1000 # g/mL		
			vol[i] = self.mw[i] / density # cm3/mol
		return vol

	@property
	def H_id_mix(self):
		Hpc = self.Hsim_pc
		# now calculate Hmix for the system
		Hpc_sum = np.zeros(self.z.shape[0])
		for i, mol in enumerate(self.unique_mols):
			Hpc_sum += self.z[:,i] * Hpc[mol]
		return Hpc_sum

	@property
	def Hmix(self):
		return self.Hsim - self.H_id_mix
	
	@property
	def S_ex(self):
		return (self.Hmix - self.G_ex) / self.T_sim
	
	@property
	def Smix(self):
		return (self.Hmix - self.G_mix_xv) / self.T_sim

	@property
	def nTdSmix(self):
		return self.G_mix_xv - self.Hmix

	@property
	def nrtl_taus(self):
		return self.fit_binary_NRTL_IP()
	
	def fit_binary_NRTL_IP(self):
		'''fit taus for NRTL if n == 2'''
		if len(self.unique_mols) != 2:
			return
		
		def NRTL_GE_fit(z, tau12, tau21):
			alpha = 0.2 # randomness factor == constant
			G12 = np.exp(-alpha*tau12/(self.Rc*self.T_sim))
			G21 = np.exp(-alpha*tau21/(self.Rc*self.T_sim))
			x1 = z[:,0]
			x2 = z[:,1]
			G_ex = -self.Rc * self.T_sim * (x1 * x2 * (tau21 * G21/(x1 + x2 * G21) + tau12 * G12 / (x2 + x1 * G12))) 
			G_id = self.Rc * self.T_sim * (x1 * np.log(x1) + x2 * np.log(x2))
			return G_ex + G_id

		self.nrtl_Gmix = self.G_mix_xv
		self.nrtl_Gmix0 = add_zeros(self.nrtl_Gmix)
		fit, pcov = curve_fit(NRTL_GE_fit, self.z, self.nrtl_Gmix)
		tau12, tau21 = fit
		
		np.savetxt(f"{self.kbi_method_dir}NRTL_taus_{self.kbi_method.lower()}.txt", [tau12, tau21], delimiter=",") 
		nrtl_taus = {"tau12": tau12, "tau21": tau21}
		return nrtl_taus
	

	def fit_FH_chi(self):
		'''fit chi parameter for flory-huggins if n == 2'''
		if len(self.unique_mols) != 2:
			# check that system is binary, else don't run
			return
			
		phi = self.v[:,self.solute_loc]
		self.fh_phi = phi

		N0 = self.properties_by_molid["n_electrons"].to_numpy()

		def fh_GM(x, chi, Tx):
			return 8.314E-3 * Tx * (x * np.log(x)/N0[0] + (1-x) * np.log(1-x)/N0[1]) + chi*x*(1-x)

		GM_fit_Tsim = partial(fh_GM, Tx=self.T_sim)
		
		fit, pcov = curve_fit(GM_fit_Tsim, xdata=phi, ydata=self.G_mix_xx)
		chi = fit[0]
		
		self.fh_chi = chi
		self.fh_Gmix = fh_GM(phi, chi, self.T_sim)

		with open(f'{self.kbi_method_dir}FH_chi_{self.kbi_method.lower()}.txt', 'w') as f:
			f.write(f'{self.fh_chi}\n')

	@property
	def uniquac_du(self):
		return self.fit_UNIQUAC_IP()

	def fit_UNIQUAC_IP(self):
		'''fit du interaction parameter to Hmix (for any number of components)'''
		self.r, self.q = UNIQUAC_RQ(self.smiles)
		du = fit_du_to_Hmix(z=self.z, Hmix=self.Hmix, T=self.T_sim, smiles=self.smiles)
		np.savetxt(f"{self.kbi_method_dir}UNIQUAC_du_{self.kbi_method.lower()}.txt", du, delimiter=",") 
		return du

	@property
	def z_plot(self):
		'''z-matrix for thermoydnamic model evaluations'''
		num_pts = {2:10, 3:7, 4:6, 5:5, 6:4}
		point_disc = PointDisc(num_comp=len(self.smiles), recursion_steps=num_pts[len(self.smiles)], load=True, store=False)
		z_arr = point_disc.points_mfr[1:-1,:]
		return z_arr 

	@property
	def uniquac_Hmix(self):
		try:
			self.uniquac_du
		except:
			self.fit_UNIQUAC_IP()

		uniq_model = UNIQUAC(z=self.z_plot, smiles=self.smiles, IP=self.uniquac_du)
		return uniq_model.GE_res(self.T_sim)

	@property
	def uniquac_Smix(self):
		try:
			self.uniquac_du
		except:
			self.fit_UNIQUAC_IP()

		uniq_model = UNIQUAC(z=self.z_plot, smiles=self.smiles, IP=self.uniquac_du)
		return uniq_model.GE_comb(self.T_sim) + uniq_model.Gid(self.T_sim)

	@property
	def unifac_Hmix(self):
		unif_model = UNIFAC(z=self.z_plot, T=self.T_sim, smiles=self.smiles, version="lle")
		return unif_model.Hmix()

	@property
	def unifac_Smix(self):
		unif_model = UNIFAC(z=self.z_plot, T=self.T_sim, smiles=self.smiles, version="lle")
		return unif_model.Smix()
	
	@property
	def quartic_model(self):
		quar_model = QuarticModel(z_data=self.z, z=self.z_plot, Hmix=self.Hmix, Sex=self.S_ex, molar_vol=self.molar_vol)
		return quar_model

	@property
	def quartic_Hmix(self):
		return self.quartic_model.Hmix_func(self.T_sim)

	@property
	def quartic_nTSex(self):
		return self.quartic_model.nTSex_func(self.T_sim)
	
	@property
	def quartic_Smix(self):
		return self.quartic_model.Smix_func(self.T_sim)


	

						







						



