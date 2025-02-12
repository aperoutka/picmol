import numpy as np
import pandas as pd
import glob
from scipy.integrate import trapz
from scipy.optimize import curve_fit
import os, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
plt.style.use(Path(__file__).parent / 'presentation.mplstyle')

from scipy.constants import R, pi

from .get_molecular_properties import load_molecular_properties
from .plotter import KBIPlotter
from .models.uniquac import UNIQUAC_R, UNIQUAC_Q
from .conversions import mol2vol
from .models.fh import FH

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

	def __init__(self, T_sim: int, prj_path: str, rdf_dir: str = "rdf_files", kbi_method: str = "adj", kbi_fig_dirname: str = "kbi_analysis"):

		# assumes folder organization is: project / systems / rdfs
		self.prj_path = prj_path
		# get kbi method: raw, adj, kgv
		self.kbi_method = kbi_method
		self.kbi_fig_dirname = kbi_fig_dirname
		# setup other folders
		self.setup_kbi_folders()
		# rdfs need to be located in their corresponding system folder in a subdirectory
		# assumes that there is only 1 rdf file with "mol1" and "mol2" in filename
		# assumes that rdf files are stored in a text file type (i.e., can be loaded with np.loadtxt) with x=r and y=g
		self.rdf_dir = rdf_dir
		# temperature that simulations were performed at (Kelvin)
		self.T_sim = T_sim
		# write simulation temperature file
		self.write_Tsim() 

		# for folder to be considered a system, it needs to have a .top file
		self.systems = [sys for sys in os.listdir(self.prj_path) if os.path.isdir(os.path.join(self.prj_path,sys)) and f"{sys}.top" in os.listdir(f"{self.prj_path}/{sys}/")]
		# sort systems so in order by solute moleclule number
		self.sort_systems()
		# get number of systems in project
		self.n_sys = len(self.systems)

		self.Rc = R / 1000 # kJ / mol K

		self.pure_component_dir = f"/Users/b324115/Library/CloudStorage/Box-Box/critical_phenomena_saxs/allisons_data/kbi_thermo/pure_components/"


	def run_kbi_analysis(self):
		# run kbi analysis and create plots
		self.kbi_analysis() # get kbis
		self.save_thermo_analysis() # save thermo analysis to csv
		# # make kbi plots
		kbi_plotter = KBIPlotter(self)
		kbi_plotter.make_figures()


	def write_Tsim(self):
		'''create txt file with simulation temperature'''
		with open(f'{self.kbi_dir}Tsim.txt', 'w') as f:
			f.write(f'{self.T_sim}\n')

	def setup_kbi_folders(self):
		'''create folders for kbi analysis'''
		mkdr(f"{self.prj_path}figures/")
		self.kbi_dir = mkdr(f"{self.prj_path}/figures/{self.kbi_fig_dirname}/")
		self.kbi_indiv_fig_dir = mkdr(f"{self.kbi_dir}/indiv_kbi/")
		self.kbi_indiv_data_dir = mkdr(f"{self.kbi_dir}/kbi_data/")
		self.kbi_method_dir = mkdr(f"{self.kbi_dir}/{self.kbi_method}/")

	def read_top(self, sys_parent_dir, sys):
		sys_mols = []
		sys_total_num_mols = 0
		sys_mol_nums_by_component = {}
		with open(f"{sys_parent_dir}{sys}/{sys}.top", "r") as top:
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
		try:
			self.top_info
		except AttributeError:
			self.extract_sys_info_from_top()
		return self.top_info["unique_mols"]
		
	@property
	def mol_nums_by_component(self):
		try:
			self.top_info
		except AttributeError:
			self.extract_sys_info_from_top()
		return self.top_info["mol_nums_by_component"]
	
	@property
	def total_num_mols(self):
		try:
			self.top_info
		except AttributeError:
			self.extract_sys_info_from_top()
		return self.top_info["total_num_mols"]
	
	@property
	def properties_by_molid(self):
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
	def smiles(self):
		return self.properties_by_molid["smiles"].values

	def assign_solute_mol(self):
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
		try:
			self.solute_mol
		except AttributeError: 
			self.assign_solute_mol()
		return self.solute_mol
	
	@property
	def solute_loc(self):
		return np.where(self.unique_mols==self.solute)[0][0]
	
	@property
	def solute_name(self):
		return self.mol_name_dict[self.solute]
	
	def sort_systems(self):
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
			npt_edr_files = [file for file in os.listdir('.') if f"{sys}_npt" and "edr" in file]
			if len(npt_edr_files) > 1:
				npt_edr_file = f"{sys}_npt.edr"
				sample_time = 200000
			else:
				npt_edr_file = npt_edr_files[0]
				sample_time = 3000
			
			# get box volume in simulation
			# if os.path.exists('volume.xvg') == False:
			os.system(f"echo volume | gmx energy -f {npt_edr_file} -o volume.xvg")
			time, V = np.loadtxt('volume.xvg', comments=["#", "@"], unpack=True)
			start_ind = np.abs(time - sample_time).argmin()
			box_vol_nm3 = np.mean(V[start_ind:])

			# get simulation enthalpy
			if os.path.exists('enthalpy.xvg') == False:
				os.system(f"echo enthalpy | gmx energy -f {npt_edr_file} -o enthalpy.xvg")
			time, H = np.loadtxt('enthalpy.xvg', comments=["#", "@"], unpack=True)
			start_ind = np.abs(time - sample_time).argmin()
			Hsim_kJ = np.mean(H[start_ind:])/self.total_num_mols[sys]

			# add properties to dataframe
			df_comp.loc[s, "systems"] = sys
			for m, mol in enumerate(mols_present):
				df_comp.loc[s, f"x_{mol}"] = self.mol_nums_by_component[sys][mol] / self.total_num_mols[sys]
				df_comp.loc[s, f"phi_{mol}"] = (self.mol_nums_by_component[sys][mol] * self.properties_by_molid["molar_vol"][mol]) / box_vol_nm3 * (10**21) / (6.022E23)
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
		filter = (r >= r_lo) & (r <= r_hi)
		r_filt = r[filter]
		g_filt = g[filter]

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
						rdf_file = glob.glob(f"{self.prj_path}{sys}/{self.rdf_dir}/*{mol_1}*{mol_2}*")[0]
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
						kbi_cm3_mol_r = np.zeros((len(r)))
						kbi_nm3_r = np.zeros((len(r)))
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
	def z0(self):
		z0 = np.zeros((self.z.shape[0]+2, self.z.shape[1]))
		z0[0,:] = [0,1]
		z0[1:-1,:] = self.z
		z0[-1,:] = [1,0]
		return z0

	@property
	def v(self):
		try:
			self.df_comp
		except AttributeError: 
			self.system_compositions()
		v_mat = np.empty((self.df_comp.shape[0], len(self.unique_mols)))
		for i, mol in enumerate(self.unique_mols):
			v_mat[:,i] = self.df_comp[f'phi_{mol}'].to_numpy()
		return v_mat
	
	@property
	def phis(self):
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
	
	@property
	def gammas(self):
		'''numerical integration of activity coefs. '''		
		dlny = self.dlngamma_dxs
		int_dlny_dx = np.zeros((self.z.shape[0], 2, self.z.shape[1]))
		for i, mol in enumerate(self.unique_mols):
			x = self.z[:,i]
			dlnyi = dlny[:,i]
			# set up array
			int_arr = np.zeros((self.z.shape[0]+1,3))
			int_arr[:-1,0] = x
			int_arr[:-1,1] = dlnyi
			int_arr[-1,:2] = [1,0]
			# sort into descending order based on mol frac
			sorted_inds = np.argsort(int_arr[:,0])[::-1]
			int_arr = int_arr[sorted_inds]

			# now perform calculation
			y0=0
			for j in range(1, self.z.shape[0]+1):
				if j > 1:
					y0 = int_arr[j-1, 2]
				# uses midpoint rule, ie., trapezoid method for numerical integration
				int_arr[j,2] = y0 + 0.5*(int_arr[j,1]+int_arr[j-1,1])*(int_arr[j,0]-int_arr[j-1,0])

			x0_ind = np.where(int_arr[:,0] == 1.)[0][0]
			int_arr = np.delete(int_arr, x0_ind, axis=0)
			# if starting at small x, flip back
			if x[0] < x[-1]:
				int_arr = int_arr[::-1]

			int_dlny_dx[:,:,i] = np.array([self.z[:,0], np.exp(int_arr[:,2])]).T
		
		gammas = int_dlny_dx[:,1,:]
		return gammas

	@property
	def G_ex(self):
		return self.Rc * self.T_sim * (np.log(self.gammas) * self.z).sum(axis=1)
	
	def G_id(self, x1, x2):
		x_sum = np.zeros(self.z.shape[0])
		phi_sum = np.zeros(self.z.shape[0])
		gid = np.zeros(self.z.shape[0])
		for i in range(len(self.unique_mols)-1):
			x_sum += x1[:,i]
			phi_sum += x2[:,i]
			gid += (x1[:,i] * np.log(x2[:,i]))
		gid += (1-x_sum) * np.log(1-phi_sum)
		gid *= self.Rc * self.T_sim
		return gid
	
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
			try:
				os.chdir(f"{self.pure_component_dir}/{sys}/")
				mols_present, total_num_mols, mol_nums_by_component = self.read_top(sys_parent_dir=self.pure_component_dir, sys=sys)
				# get npt edr files for system properties; volume, enthalpy.
				npt_edr_files = [file for file in os.listdir('.') if f"{sys}_npt" and "edr" in file]
				if len(npt_edr_files) > 1:
					npt_edr_file = f"{sys}_npt.edr"
				else:
					npt_edr_file = npt_edr_files[0]
				# get simulation enthalpy
				if os.path.exists('enthalpy.xvg') == False:
					os.system(f"echo enthalpy | gmx energy -f {npt_edr_file} -o enthalpy.xvg")
				time, H = np.loadtxt('enthalpy.xvg', comments=["#", "@"], unpack=True)
				H_pc[mol] = np.mean(H)/total_num_mols
			except:
				# if file/path does not exist just use nan values
				H_pc[mol] = np.nan
		return H_pc

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
		return (self.Hmix - self.G_mix_xx) / self.T_sim
	
	@property
	def Gex0(self):
		return add_zeros(self.G_ex)
		
	def save_thermo_analysis(self):
		try:
			self.df_comp
		except AttributeError: 
			self.system_compositions()
		df_thermo = pd.DataFrame()
		for i, mol in enumerate(self.unique_mols):
			df_thermo[f"x_{mol}"] = self.z0[:,i]
		for i, mol in enumerate(self.unique_mols):
			df_thermo[f"phi_{mol}"] = self.v0[:,i]
		for i, mol in enumerate(self.unique_mols):
			n_rows = self.z.shape[0]
			df_thermo.loc[1:n_rows, f"gamma_{mol}"] = self.gammas[:,i]
		df_thermo["G_ex"] = add_zeros(self.G_ex)
		for xx in ['xx', 'xv', 'vv']:
			df_thermo[f"G_mix_{xx}"] = add_zeros(getattr(self, f"G_mix_{xx}"))
		df_thermo["H_sim"] = add_zeros(self.Hsim)
		df_thermo["H_id_mix"] = add_zeros(self.H_id_mix)
		df_thermo["H_mix"] = add_zeros(self.Hmix)
		df_thermo["S_ex"] = add_zeros(self.S_ex)
		df_thermo["S_mix"] = add_zeros(self.Smix)
		df_thermo.to_csv(f"{self.kbi_method_dir}thermo_analysis_{self.kbi_method.lower()}.csv", index=False)



###
	# fitting interaction paramters for NRTL, UNIQUAC, FH
	def fit_binary_NRTL_IP(self):
		if len(self.unique_mols) != 2:
			# check that system is binary, else don't run
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

		self.nrtl_Gmix = self.G_mix_xx
		self.nrtl_Gmix0 = add_zeros(self.G_mix_xx)
		fit, pcov = curve_fit(NRTL_GE_fit, self.z, self.nrtl_Gmix)
		tau12, tau21 = fit
		
		np.savetxt(f"{self.kbi_method_dir}NRTL_taus_{self.kbi_method.lower()}.txt", [tau12, tau21], delimiter=",") 
		self.nrtl_taus = {"tau12": tau12, "tau21": tau21}

	def fit_UNIQUAC_IP(self):

		N_mols = len(self.unique_mols)
		self.r = UNIQUAC_R(list(self.mol_smiles_dict.values()))
		self.q = UNIQUAC_Q(list(self.mol_smiles_dict.values()))
		globals()['r'] = self.r
		globals()['q'] = self.q
		globals()['Rc'] = self.Rc
		globals()['T_sim'] = self.T_sim

		def create_du_function(ij):
				# Validate inputs
				if not isinstance(ij, int):
					raise TypeError("ij must be integer.")

				# Generate argument names
				du_arg_names = [f'du{k}' for k in range(ij)]
				# Include 'z' at the beginning of the argument list
				arg_list = ', '.join(['z'] + du_arg_names)  # e.g., 'z, du0, du1, du2'

				# Build the function code as a string
				func_code = f"""
def UNIQUAC_GM_fit_func({arg_list}):
				\"""
    		Processes 'z' and 'du' arguments in range du_{ij}.

				Parameters:
				z : numeric
		{chr(10).join(['    {} : numeric'.format(name) for name in du_arg_names])}
				\"""

				# Collect 'du' arguments into a list
				du_list = [{', '.join(du_arg_names)}]
				N_mols = z.shape[1]

				du = np.zeros((N_mols, N_mols))

				ij = 0
				for i in range(N_mols):
					for j in range(N_mols):
						if i < j:
							du[i,j] = du_list[ij]
							du[j,i] = du_list[ij]
							ij += 1
				
				r = globals()['r']
				q = globals()['q']
				Rc = globals()['Rc']
				T_sim = globals()['T_sim']

				zc = 10
				l = (zc / 2) * (r - q) - (r - 1)
				tau = np.exp(-du / (Rc * T_sim))
				rbar = z @ r
				qbar = z @ q
				# Calculate phi and theta (segment and area fractions)
				phi = z * (1 / rbar[:, None]) * r
				theta = z * (1 / qbar[:, None]) * q
				# Solve for free energy of mixing GM 
				GM_comb = Rc * T_sim * (np.sum(z * np.log(phi), axis=1) + (zc / 2) * (z * (np.log(theta) - np.log(phi))) @ q)
				GM_res = -Rc * T_sim * (z * np.log(np.dot(theta , tau))) @ q
				return GM_comb + GM_res
"""
				# Create a local namespace for the function definition
				func_namespace = {}
				# Include '__name__' in the globals dictionary
				exec(func_code, globals(), func_namespace)
				# Retrieve the dynamically created function
				du_function = func_namespace['UNIQUAC_GM_fit_func']
				return du_function
		
		ij_combo = 0
		for i in range(N_mols):
			for j in range(N_mols):
				if i < j:
					ij_combo += 1

		UNIQUAC_GM_fit = create_du_function(ij_combo)
		self.uniquac_Gmix = self.G_mix_xx
		self.uniquac_Gmix0 = add_zeros(self.G_mix_xx)
		popt, pcov = curve_fit(UNIQUAC_GM_fit, self.z, self.uniquac_Gmix)

		du = np.zeros((len(self.unique_mols), len(self.unique_mols)))
		ij = 0
		for i in range(len(self.unique_mols)):
			for j in range(len(self.unique_mols)):
				if i < j:
					du[i,j] = popt[ij]
					du[j,i] = popt[ij]
					ij += 1
			
		np.savetxt(f"{self.kbi_method_dir}UNIQUAC_du_{self.kbi_method.lower()}.txt", du, delimiter=",") 
		self.uniquac_du = du


	def fit_FH_chi(self):
		solute_loc = np.where(self.unique_mols==self.solute)[0][0]
		phi = self.phis[:,solute_loc]
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

		

						







						



