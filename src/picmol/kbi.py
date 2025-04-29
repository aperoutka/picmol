from re import L
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

from scipy.constants import R, pi, N_A

from .get_molecular_properties import load_molecular_properties
from .models.uniquac import UNIQUAC_RQ, fit_du_to_Hmix
from .models import UNIQUAC, UNIFAC, QuarticModel, FH
from .models.cem import PointDisc
from .functions import get_solute_molid, mol2vol


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
	r"""
	Class for calculating Kirkwood-Buff Integrals from radial distribution functions in a GROMACS project

	:param prj_path: absolute path to project directory; contains subdirectories containing systems as a function of composition
	:type prj_path: str
	:param pure_component_path: absolute path to pure component directory; contains subdirectories of molecules at a given temperature
	:type pure_component_path: str
	:param rdf_dir: directory name where RDF files are located in each system directory, default is `rdf_files`
	:type rdf_dir: str, optional
	:param kbi_method: correction method to apply to correlation function, default is `adj`
	:type kbi_method: str, optional
	:param kbi_method options:
		* 'raw': no correction
		* 'adj': correcting for tail of RDF :math:`\neq` 1 at large r
		* 'gv': correcting the number densities to account for excess/depletion, introduced by Ganguly and Van der Vegt
		* 'kgv': applying a damping function introduced by Kruger to the correction function by Ganguly and Van der Vegt
	:param rkbi_min: fraction of r; lower bound for extrapolation to the thermodynamic limit, default is 0.75. float will be applied to all systems and dict uses systems as keys with corresponding float values.
	:type rkbi_min: float or dict, optional
	:param kbi_fig_dirname: directory name for results from KBI analysis, default is `kbi_analysis`
	:type kbi_fig_dirname: str, optional
	:param avg_start_time: start time (ns) for property (temperature, volume, enthalpy) analysis from .edr file, default is 100
	:type avg_start_time: float, optional
	:param avg_end_time: end time (ns) for property (temperature, volume, enthalpy) analysis from .edr file, default is end of trajectory
	:type avg_start_time: float, optional
	:param solute_mol: molecule name in .top file corresponding to molecule used to sort systems
	:type solute_mol: str
	:param geom_mean_pairs: pairs of molecules for taking the geometric mean of, default is empty list
	:type geom_mean_pairs: list[list[molecule_ids]], optional
	"""
	def __init__(
			self, 
			prj_path: str, 
			pure_component_path: str,
			rdf_dir: str = "rdf_files", 
			kbi_method: str = "adj",
			rkbi_min = 0.75,
			kbi_fig_dirname: str = "kbi_analysis",
			avg_start_time = 100, 
			avg_end_time = None,
			solute_mol = None,
			geom_mean_pairs = [],
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

		# get min value for kbi extrapolation
		# this is the minimum value of rkbi to use for extrapolation; this should be set based on the system being studied
		# default value, 0.5 -> start at 1/2 max(r) in rdf
		# if not float, use dict to assign value for each system
		self.rkbi_min = rkbi_min

		# geom mean pair should be a list of lists, i.e., which molecules together should be represented with a geometric mean rather than their pure components --> applied after pure component activity coefficient calculation
		self.geom_mean_pairs = geom_mean_pairs
		
		# get kbi method: raw, adj, kgv
		self.kbi_method = kbi_method.lower()
		self.kbi_fig_dirname = kbi_fig_dirname

		# setup other folders
		self._setup_kbi_folders()

		# rdfs need to be located in their corresponding system folder in a subdirectory
		# assumes that there is only 1 rdf file with "mol1" and "mol2" in filename
		# assumes that rdf files are stored in a text file type (i.e., can be loaded with np.loadtxt) with x=r and y=g
		self.rdf_dir = rdf_dir

		# for folder to be considered a system, it needs to have a .top file
		self.systems = [sys for sys in os.listdir(self.prj_path) if os.path.isdir(os.path.join(self.prj_path,sys)) and f"{sys}.top" in os.listdir(f"{self.prj_path}/{sys}/")]

		# get number of systems in project
		self.n_sys = len(self.systems)

		# get gas constant in kJ/mol-K
		self.Rc = R / 1000 # kJ / mol K
		self.N_A = N_A

		# initialize properties for KBI analysis
		self._unique_mols = self._top_unique_mols

		# specifiy solute mol if desired, otherwise preference will be given to order of separation systems, i.e., extractant > modifier > solute (ie., water) > solvent
		self._solute = solute_mol
		if self._solute is None:
			self._solute = get_solute_molid(self._top_unique_mols, self.mol_class_dict)
		self._top_solute = self._solute # get initial solute
		self._top_solute_loc = self.solute_loc # get idx of initial solute

		# sort systems so in order by solute moleclule number
		self._sort_systems()

		self._mol_name_dict = {mol: self.properties_by_molid["mol_name"][mol] for mol in self.unique_mols}
		self._z = self._top_z # get mol fraction matrix of each system
		self._v = self._top_v # convert mol fraction matrix to vol fraction
	
	def run(self):
		"""
		Run KBI analysis and calculate activity coefficients and Gibbs thermodynamic properties
		"""
		# run the KBI analysis
		self._calculate_kbi()
		# calculate thermodynamic properties
		self.gammas() # this needs to be run to ensure geometric means are taken into account for excess property calculation.
		self.GM() # makes sure that all properties get run for Gibbs energy

	def _get_edr_file(self, sys):
		"""
		Get the .edr file for a given system, requires that 'npt' is in filename to distinguish from other ensembles. This is used to get the temperature, volume, enthalpy.

		:param sys: system name
		:type sys: str
		:return: edr filename
		:rtype: str
		"""
		npt_edr_files = [file for file in os.listdir('.') if (sys in file) and ("npt" in file) and ("edr" in file)]
		if len(npt_edr_files) > 1:
			for file in npt_edr_files:
				if 'init' not in file and 'eqm' not in file:
					return file
		else:
			return npt_edr_files[0]


	def _get_time_average(self, time, arr):
		"""
		Get time average from property using .edr file

		:param time: array of times that property was calculated over
		:type time: np.array
		:param arr: property calculated from .edr file
		:type arr: np.array
		:return: average of property from `avg_start_time` to `avg_end_time`
		:rtype: float
		"""
		start_ind = np.abs(time - self.avg_start_time).argmin()
		if self.avg_end_time is not None:
			end_ind = np.abs(time - self.avg_end_time).argmin()
			return np.mean(arr[start_ind:end_ind])
		else:
			return np.mean(arr[start_ind:])

	def _get_simulation_temps(self):
		"""
		Get actual simulation temperature (K) for each system

		:return: array of average simulation temperatures from `avg_start_time` to `avg_end_time`
		:rtype: np.array
		"""
		sys_Tsims = np.zeros(self.n_sys)
		for s, sys in enumerate(self.systems):
			os.chdir(f"{self.prj_path}/{sys}/")
			# get .edr file
			npt_edr_file = self._get_edr_file(sys=sys)
			# get temperature from .edr file
			if os.path.exists('temperature.xvg') == False:
				os.system(f"echo temperature | gmx energy -f {npt_edr_file} -o temperature.xvg")
			# average temperatures over time
			time, T = np.loadtxt('temperature.xvg', comments=["#", "@"], unpack=True)
			sys_Tsims[s] = self._get_time_average(time, T)
		self._sys_Tsims = sys_Tsims # save to instance variable for later use
		return self._sys_Tsims

	@property
	def _simulation_temps(self):
		try:
			self._sys_Tsims
		except AttributeError:
			self._get_simulation_temps()
		return self._sys_Tsims

	@property
	def T_sim(self):
		"""
		:return: average simulation temperature (K)
		"""
		return round(np.mean(self._simulation_temps))

	def _setup_kbi_folders(self):
		mkdr(f"{self.prj_path}/figures/")
		self.kbi_dir = mkdr(f"{self.prj_path}/figures/{self.kbi_fig_dirname}/")
		self.kbi_method_dir = mkdr(f"{self.kbi_dir}/{self.kbi_method}_kbi_method/")
		self.kbi_indiv_fig_dir = mkdr(f"{self.kbi_method_dir}/indiv_kbi/")

	def _read_top(self, sys_parent_dir, sys):
		"""
		Extract molecule names and corresponding number in .top file

		:param sys_parent_dir: name for project directory
		:type sys_parent_dir: str
		:param sys: system name
		:type sys: str
		:return: list[molecule names], total number of molecules, and dict[key: molecule name, value: molecule number]
		"""
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
	

	def _extract_sys_info_from_top(self):
		"""
		Create dictionary containing top info for each system in project
		
		:return: dictionary containing unique np.array of molecule names, total number of molecules in each system, and molecule number by component for each system
		:rtype: dict
		"""
		mols_present = []
		total_num_mols = {sys: 0 for sys in self.systems}
		mol_nums_by_component = {sys: {} for sys in self.systems}
		for s, sys in enumerate(self.systems):
			sys_mols, sys_total_num_mols, sys_mol_nums_by_component = self._read_top(sys_parent_dir=self.prj_path, sys=sys)
			for mol in sys_mols:
				if mol not in mols_present:
					mols_present.append(mol)
			total_num_mols[sys] = sys_total_num_mols
			mol_nums_by_component[sys] = sys_mol_nums_by_component
		# get unique mols in the system
		# unique_mols = np.unique(mols_present)
		self._top_info = {
			"unique_mols": np.array(mols_present), 
			"mol_nums_by_component": mol_nums_by_component, 
			"total_num_mols": total_num_mols
		}

	@property
	def _top_unique_mols(self):
		try:
			self._top_info
		except AttributeError:
			self._extract_sys_info_from_top()
		return self._top_info["unique_mols"]

	@property
	def unique_mols(self):
		"""
		:return: unique molecule names in .top file for the project
		"""
		return self._unique_mols

	@unique_mols.setter
	def unique_mols(self, value):
		self._unique_mols = value

	@property
	def num_comp(self):
		"""
		:return: number of components in project
		"""
		return len(self.unique_mols)

	@property
	def mol_nums_by_component(self):
		"""
		:return: dictionary of molecule numbers by component for each system, read from .top file
		"""
		try:
			self._top_info
		except AttributeError:
			self._extract_sys_info_from_top()
		return self._top_info["mol_nums_by_component"]
	
	@property
	def total_num_mols(self):
		"""
		:return: dictionary of total number of molecules in each system, read from .top file
		"""
		try:
			self._top_info
		except AttributeError:
			self._extract_sys_info_from_top()
		return self._top_info["total_num_mols"]
	
	@property
	def properties_by_molid(self):
		"""
		:return: pandas DataFrame only for molecules present in project. Load `molecular_properties.csv` and set molecule name in .top file as index. 
		"""
		prop_df = load_molecular_properties("mol_id")
		return prop_df.loc[self.unique_mols, :]
	
	@property
	def mol_name_dict(self):
		return self._mol_name_dict

	def add_molname_to_dict(self, mol_id, mol_name):
		if mol_id not in self._mol_name_dict.keys():
			self._mol_name_dict[mol_id] = mol_name

	@property
	def mol_class_dict(self):
		return {mol: self.properties_by_molid["mol_class"][mol] for mol in self.unique_mols}

	@property
	def mol_smiles_dict(self):
		return {mol: self.properties_by_molid["smiles"][mol] for mol in self.unique_mols}

	@property
	def molar_vol(self):
		"""
		:return: np.array of pure component molar volumes (cm3/mol), defaults to results from pure component simulations, if pure component simulation not found use value at STP from `molecular_properties.csv`
		"""
		V0 = np.zeros(len(self.unique_mols)) # initialize array for molar volumes
		for i, mol in enumerate(self.unique_mols):
			# try to get molar volume from simulation results first
			if np.isnan(self.md_molar_vol[i]):
				# fallback to experimental values
				V0[i] = self.exp_molar_vol[i]
			else:
				V0[i] = self.md_molar_vol[i]
		return V0
	
	@property
	def exp_molar_vol(self):
		return self.properties_by_molid["molar_vol"].to_numpy()
	
	@property
	def n_electrons(self):
		return self.properties_by_molid["n_electrons"].to_numpy()
	
	@property
	def mol_charge(self):
		return self.properties_by_molid["mol_charge"].to_numpy()

	@property
	def mw(self):
		return self.properties_by_molid["mol_wt"].to_numpy()
	
	@property
	def smiles(self):
		return self.properties_by_molid["smiles"].values
	
	@property
	def solute(self):
		# get solute mol_id
		return self._solute
	
	@solute.setter
	def solute(self, value):
		self._solute = value
	
	@property
	def solute_loc(self):
		# get index of solute molecule
		return self._mol_idx(mol=self.solute)
	
	@property
	def solute_name(self):
		# get name of solute molecule
		return self._mol_name_dict[self.solute]
	
	def _sort_systems(self):
		'''sort systems based on molar fraction'''
		sys_df = pd.DataFrame({
			"systems": self.systems,
			"mols": [self.mol_nums_by_component[sys][self.solute] for sys in self.systems]
		})
		sys_df = sys_df.sort_values("mols").reset_index(drop=True)
		self.systems = sys_df["systems"].to_list()

	@property
	def _box_vol_nm3(self):
		'''calculate the average box volume in nm^3 for each system'''
		vol = np.zeros(self.n_sys)
		for s, sys in enumerate(self.systems):
			# change to system directory
			os.chdir(f"{self.prj_path}/{sys}/")
			# get system volume
			if os.path.exists('volume.xvg') == False:
				os.system(f"echo volume | gmx energy -f {self._get_edr_file(sys=sys)} -o volume.xvg")
			time, V = np.loadtxt('volume.xvg', comments=["#", "@"], unpack=True)
			vol[s] = self._get_time_average(time, V)
		return vol

	@property
	def Hsim(self):
		# calculate the average enthalpy for each system
		H = np.zeros(self.n_sys)
		for s, sys in enumerate(self.systems):
			# change to system directory
			os.chdir(f"{self.prj_path}/{sys}/")
			# get system enthalpy
			if os.path.exists('enthalpy_npt.xvg') == False:
				os.system(f"echo enthalpy | gmx energy -f {self._get_edr_file(sys=sys)} -o enthalpy_npt.xvg")
			time, H_sys = np.loadtxt('enthalpy_npt.xvg', comments=["#", "@"], unpack=True)
			H[s] = self._get_time_average(time, H_sys)/self.total_num_mols[sys]
		return H
	
	@property
	def _n_mol(self):
		'''calculate molecule numbers for each component in system'''
		n_mol = np.zeros((self.n_sys, len(self._top_unique_mols))) # initialize array for molecule numbers
		for s, sys in enumerate(self.systems):
			for i, mol in enumerate(self._top_unique_mols):
				try:
					n_mol[s,i] = self.mol_nums_by_component[sys][mol]
				except:
					n_mol[s,i] = 0 # if molecule not found
		return n_mol

	@property
	def _top_z(self):
		'''get mol fraction matrix from system compositions'''
		return self._n_mol / self._n_mol.sum(axis=1)[:,np.newaxis]

	@property
	def _top_v(self):
		'''convert z_mat to vol fraction matrix'''
		return mol2vol(self._top_z, self.molar_vol)

	@property
	def _top_c(self):
		'''calculate molarity (mol/L) of each molecule in system'''
		return self._top_rho * (10**24) / N_A

	@property
	def _top_rho(self):
		'''calculate the number density for each molecule in system'''
		return self._n_mol / self._box_vol_nm3[:,np.newaxis]

	def _system_compositions(self):
		"""get system properties at each composition"""
		df_comp = pd.DataFrame()
		for s, sys in enumerate(self.systems):
			# add properties to dataframe
			df_comp.loc[s, "systems"] = sys # system name
			df_comp.loc[s, "T_sim"] = round(self._simulation_temps[s], 4) # actual simulation temperature for the system
			for i, mol in enumerate(self.unique_mols):
				df_comp.loc[s, f"x_{mol}"] = self._top_z[s,i] # mole fracation of mol i
				df_comp.loc[s, f"phi_{mol}"] = self._top_v[s,i] # volume fraction of mol i
				df_comp.loc[s, f"c_{mol}_M"] = self._top_c[s,i] # molarity of mol i (mol/L)
				df_comp.loc[s, f'rho_{mol}'] = self._top_rho[s,i] # number density of mol i (nm^-3)
				df_comp.loc[s, f'n_{mol}'] = self._n_mol[s,i] # number of molecules of mol i in the system
			df_comp.loc[s, 'n_tot'] = self._n_mol[s].sum() # total number of molecules in the system
			df_comp.loc[s, 'box_vol'] = self._box_vol_nm3[s] # box volume in nm^3 for the system
			df_comp.loc[s, 'enthalpy'] = self.Hsim[s] # average enthalpy per molecule in kJ/mol
			# replace all NaN values with zeros
			df_comp.fillna(0, inplace=True)
			# save to csv
			df_comp.to_csv(f'{self.kbi_dir}system_compositions.csv', index=False)
			self.df_comp = df_comp

	def kbi_npt(self, r, g, r_lo, r_hi, avg, sys_num, mol_i, mol_j, method):
		r"""
		For a given radial distribution function, calculate KBI from r_lo to r_hi by applying a correction to the correlation function.

		:param r: distance (nm) between two molecules in radial distribution function
		:type r: np.array
		:param g: radial distribution function for two molecules as a function of r
		:type g: np.array
		:param r_lo: minimum r (nm) to evaluate integral over
		:type r_lo: float
		:param r_hi: maximum r (nm) to evaluate integral over
		:type r_hi: float
		:param avg: average value of g as r -> R, i.e., average value of the tail of RDF
		:type avg: float
		:param sys_num: system index
		:type sys_num: int
		:param mol_i: molecule id in .top file for i in RDF, :math:`g_ij(r)`
		:type mol_i: str
		:param mol_j: molecule id in .top file for j in RDF, :math:`g_ij(r)`
		:type mol_j: str
		:param method: KBI correction method to implement
		:type method: str
		:param method options:
			* 'raw': no correction
			* 'adj': correcting for tail of RDF != 1 at large r
			* 'gv': correcting the number densities to account for excess/depletion, introduced by Ganguly and Van der Vegt
			* 'kgv': applying a damping function introduced by Kruger to the correction function by Ganguly and Van der Vegt

		
		Corrections to correlation function (:math:`h_{ij}(r)`), where :math:`g_{ij}^{NpT}` is the radial distribution function in NpT ensemble, :math:`N_j` is the number of molecules of type j, :math:`\Delta N_{ij}` is the number of molecules of type j around molecule i in radius of shell dr, and :math:`V` is the volume of simulation box.
		
		.. math::
			h_{ij}^{raw}(r) = g_{ij}^{NpT}(r) - 1

		.. math::
			h_{ij}^{adj}(r) = g_{ij}^{NpT}(r) - avg

		.. math::
			h_{ij}^{gv}(r) = g_{ij}^{NpT}(r) \left(\frac{N_j\left(1 - \frac{4/3 \pi r^3}{V} \right)}{N_j \left(1 - \frac{4/3 \pi r^3}{V} \right) - \Delta N_{ij}  - \delta_{ij}} \right) - 1

		.. math::
			\Delta N_{ij} = \int_0^R 4 \pi r^2 \rho_j \left(g_{ij}^{NpT}(r) - 1\right) dr

		.. math::
			h_{ij}^{kgv}(r) = h_{ij}^{gv} \left(1 - \frac{3r}{2R} + \frac{r^3}{2r^3} \right)


		KBI (:math:`G_{ij}^R`) calculations from corrected correlation functions (:math:`h_{ij}(r)`).

		.. math::
			G_{ij}^R =	\int_0^R 4 \pi r^2 h_{ij}(r) dr

		:return: KBI in units of nm3 and cm3/mol.
		"""
		# get max r
		r_max = max(r)

		# filter r and g(r)
		r_filter = (r >= r_lo) & (r <= r_hi)
		r_filt = r[r_filter]
		g_filt = g[r_filter]

		# get molecular properties
		rho_moli = float(self.df_comp.loc[sys_num,f'rho_{mol_i}'])
		rho_molj = float(self.df_comp.loc[sys_num,f'rho_{mol_j}'])
		c_moli = float(self.df_comp.loc[sys_num, f'c_{mol_i}_M'])
		Nj = int(self.df_comp.loc[sys_num, f'n_{mol_j}'])
		box_vol = float(self.df_comp.loc[sys_num, f'box_vol'])
		kd = int(mol_i == mol_j)

		# get spacing between datapoints
		dr = 0.002 # GROMACS default
		# adjustment: ie., correct for g(r) != 1 at R
		if method.lower() == 'adj':
			h = g_filt - avg
		# no correction
		elif method.lower() == 'raw':
			h = g_filt - 1
		# for no damping
		elif method.lower() in ['gv', 'kgv']:
			# 1-volume ratio
			vr = 1 - ((4/3)*pi*r_filt**3/box_vol) 
			# coordination number of mol_2 surrounding mol_1
			cn = 4 * pi * r_filt**2 * rho_molj * (g_filt - 1)
			dNij = trapz(cn, x=r_filt, dx=dr)	
			# g(r) correction using Ganguly - van der Vegt approach
			g_gv_correct = g_filt * Nj * vr / (Nj * vr - dNij - kd) 
			h = g_gv_correct - 1
			# apply damping function
		if 'kgv' in method.lower() or 'k' in method.lower():
			# combo of g(r) correction with damping function K. 
			damp_k = (1 - (3*r_filt)/(2*r_max) + r_filt**3/(2*r_max**3))
			h *= damp_k
		
		f = 4 * pi * r_filt**2 * h
		kbi_nm3 = trapz(f, x=r_filt, dx=dr)
		kbi_cm3_mol = kbi_nm3 * rho_moli * 1000 / c_moli
		return kbi_nm3, kbi_cm3_mol

	def kbi_thermo_limit(self, L, rkbi, min_L_idx):
		r"""
		Extrapolate KBI to the thermodynamic limit.

		:param L: fraction of linear dimension of box volume
		:type L: np.array
		:param rkbi: KBI as a function of r in RDF, i.e., running KBI
		:type rkbi: np.array
		:param min_L_idx: lower bound of L for extrapolation
		:type min_L_idx: float

		.. math::
			\lambda G_{ij}^R = \lambda G_{ij}^{\infty} + F_{ij}^{\infty}
		
		:return: list[:math:`G_{ij}^{\infty}`, :math:`F_{ij}^{\infty}`]
		"""
		'''extrapolate kbi values to the thermodynamic limit'''
		x = L 
		y = L * rkbi 
		x_fit = x[min_L_idx:]
		y_fit = y[min_L_idx:]

		def fGij_inf(l, Gij, b):
			# function to fit Gij_R to the infinite dilution limit; this is used for curve fitting
			return Gij*l + b

		params, pcov = curve_fit(fGij_inf, xdata=x_fit, ydata=y_fit)
		return params

	def _calculate_kbi(self):
		'''calculated KBI values from rdf files'''
		try:
			self.df_comp
		except AttributeError: 
			self._system_compositions()
		
		# create dataframes for each pairwise interaction
		df_kbi = pd.DataFrame()
		df_kbi[f"x_{self.solute}"] = self.df_comp[f"x_{self.solute}"]
		df_kbi[f"phi_{self.solute}"] = self.df_comp[f"phi_{self.solute}"]
		for i, mol_1 in enumerate(self.unique_mols):
			for j, mol_2 in enumerate(self.unique_mols):
				if i <= j:
					df_kbi[f'G_{mol_1}_{mol_2}_nm3'] = np.zeros(self.n_sys)
					df_kbi[f'G_{mol_1}_{mol_2}_cm3_mol'] = np.zeros(self.n_sys)
		
		# create dict fo storing inf fits
		self.kbi_inf_fits = {sys: {} for sys in self.systems}
		# storing system lambda values
		self.lamdba_values = {sys: {} for sys in self.systems}
		self.lamdba_values_fit = {sys: {} for sys in self.systems}

		for s, sys in enumerate(self.systems):
			# create kbi dataframe for each system for storing kbi's as a function of r
			df_kbi_sys = pd.DataFrame()
			# iterate through all possible molecular combinations with no repeats
			for i, mol_1 in enumerate(self.unique_mols):
				for j, mol_2 in enumerate(self.unique_mols):
					if i <= j:
						# check that both molecules are in system
						if (mol_1 not in self.mol_nums_by_component[sys].keys()) or (mol_2 not in self.mol_nums_by_component[sys].keys()):
							continue
						try:
							rdf_file = glob.glob(f"{self.prj_path}/{sys}/{self.rdf_dir}/*{mol_1}*{mol_2}*")[0]
						except:
							rdf_file = glob.glob(f"{self.prj_path}/{sys}/{self.rdf_dir}/*{mol_2}*{mol_1}*")[0]

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
						kbi_cm3_mol_r = np.full((len(r)-1), fill_value=np.nan)
						kbi_nm3_r = np.full((len(r)-1), fill_value=np.nan)
						kbi_cm3_mol_sum = 0.
						kbi_nm3_sum = 0.
						for k in range(len(r)-1):
							kbi_nm3, kbi_cm3_mol = self.kbi_npt(r=r, g=g, r_lo=r[k], r_hi=r[k+1], sys_num=s, avg=limit_g_not_1, mol_i=mol_1, mol_j=mol_2, method=self.kbi_method)
							kbi_cm3_mol_sum += kbi_cm3_mol
							kbi_nm3_sum += kbi_nm3
							kbi_cm3_mol_r[k] = kbi_cm3_mol_sum
							kbi_nm3_r[k] = kbi_nm3_sum
						
						# calculate kbis in thermodynamic limit
						V_cell = (4/3)*pi*r[:-1]**3 # volume of the spherical cell (for the integration)
						L = (V_cell/V_cell.max())**(1/3)
						if type(self.rkbi_min) == dict:
							sys_rkbi_min = self.rkbi_min[sys]
						else:
							sys_rkbi_min = self.rkbi_min
						min_L_idx = np.abs(r[:-1]/r.max() - sys_rkbi_min).argmin() # find the index of the minimum L value to start extrapolation
						params_nm3 = self.kbi_thermo_limit(L=L, rkbi=kbi_nm3_r, min_L_idx=min_L_idx)
						Gij_inf_nm3, _ = params_nm3
						params_cm3_mol = self.kbi_thermo_limit(L=L, rkbi=kbi_cm3_mol_r, min_L_idx=min_L_idx)
						Gij_inf_cm3_mol, _ = params_cm3_mol

						self.kbi_inf_fits[sys][f'{mol_1}-{mol_2}'] = np.poly1d(params_cm3_mol)
						self.lamdba_values[sys][f'{mol_1}-{mol_2}'] = L
						self.lamdba_values_fit[sys][f'{mol_1}-{mol_2}'] = L[min_L_idx:]

						# add kbi values to nested dictionaries
						df_kbi.loc[s, f'G_{mol_1}_{mol_2}_nm3'] = Gij_inf_nm3
						df_kbi.loc[s, f'G_{mol_1}_{mol_2}_cm3_mol'] = Gij_inf_cm3_mol

						# save to dataframe for plotting purposes
						df_kbi_sys["r"] = r[:-1]
						df_kbi_sys[f'G_{mol_1}_{mol_2}_nm3'] = kbi_nm3_r
						df_kbi_sys[f'G_{mol_1}_{mol_2}_cm3_mol'] = kbi_cm3_mol_r
						setattr(self, f'kbi_{s}', df_kbi_sys)

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
		r"""
		Mol fraction array (:math:`z`) with elements, :math:`x_i`.

		.. math::
			x_i = \frac{N_i}{\sum_j N_j}
		
		:return: mol fraction array, with shape(:math:`len(\text{systems}) \times n`)
		"""
		return self._z

	@z.setter
	def z(self, value):
		self._z = value

	@property
	def v(self):
		r"""
		Volume fraction array (:math:`v`) with elements, :math:`\phi_i` where :math:`V_i` is molar volume of molecule of type i.

		.. math::
			\phi_i = \frac{x_i V_i}{\sum_j x_j V_j}
		
		:return: volume fraction array, with shape(:math:`len(\text{systems}) \times n`)
		"""
		return self._v
	
	@v.setter
	def v(self, value):
		self._v = value

	def G_matrix(self):
		r"""
		Construct a symmetric matrix (:math:`\textbf{G}`) of KBI values in the thermodynamic limit

		.. math::
			\textbf{G} = \begin{bmatrix}
				G_{11}^\infty & G_{12}^\infty & \cdots & G_{1n}^\infty \\
				G_{21}^\infty & G_{22}^\infty & \cdots & G_{2n}^\infty \\
				\vdots & \vdots & \ddots & \vdots \\
				G_{n1}^\infty & G_{n2}^\infty & \cdots & G_{nn}^\infty \\
			\end{bmatrix}

    where :math:`G_{ij}^\infty` is the KBI in the thermodynamic limit in nm3 between molecules of type i and j, with :math:`G_{ij}^\infty = G_{ji}^\infty`.
		
		:return: np.array of KBI values for each pairwise interaction, with :math:`shape(\textbf{G}) = len(\text{systems}) \times n \times n`
		"""
		''' create a symmetric matrix from KBI values '''
		try:
			self.df_kbi
		except AttributeError:
			self._calculate_kbi()
		try:
			self.df_comp
		except AttributeError: 
			self._system_compositions()
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
	
	def B_matrix(self):
		r"""
		Construct a symmetric matrix (:math:`\textbf{B}`) of number fluctuations from KBI values in :math:`\textbf{G}`, with elements :math:`\textbf{B}_{ij}`.

		.. math::
			\textbf{B}_{ij} = \rho_i \rho_j \textbf{G}_{ij} + \rho_i \delta_{ij}

		.. math::
			\textbf{B} = \begin{bmatrix}
				\rho_1^2 \textbf{G}_{11} + \rho_1 & \rho_1 \rho_2 \textbf{G}_{12} & \cdots & \rho_1 \rho_n \textbf{G}_{1n} \\
				\rho_1 \rho_2 \textbf{G}_{21} & \rho_2^2 \textbf{G}_{22} + \rho_2 & \cdots & \rho_2 \rho_n \textbf{G}_{2n} \\
				\vdots & \vdots & \ddots & \vdots \\
				\rho_1 \rho_n \textbf{G}_{n1} & \rho_2 \rho_n \textbf{G}_{n2} & \cdots & \rho_n^2 \textbf{G}_{nn} + \rho_n \\
			\end{bmatrix}
		
		:return: np.array with :math:`shape(\textbf{B}) = len(\text{systems}) \times n \times n`
		"""
		B = np.full((self.z.shape[0],len(self.unique_mols), len(self.unique_mols)), fill_value=np.nan)
		for i, mol_1 in enumerate(self.unique_mols):
			rho_i = self.df_comp[f'rho_{mol_1}'].to_numpy()
			for j, mol_2 in enumerate(self.unique_mols):
				rho_j = self.df_comp[f'rho_{mol_2}'].to_numpy()
				kd_ij = int(i==j)
				B[:,i,j] = rho_i * rho_j * self.G_matrix()[:,i,j] + rho_i * kd_ij
		return B

	@property
	def _B_inv(self):
		'''get inverse of matrix B'''
		return np.linalg.inv(self.B_matrix())

	@property 
	def _B_det(self):
		'''get determinant of matrix B'''
		return np.linalg.det(self.B_matrix())
	
	def cofactors_Bij(self):
		r"""
		Get the cofactors of :math:`\textbf{B}`, where elements are defined by :math:`\textbf{B}^{ij}`.
		
		.. math::
			\textbf{B}^{ij} = \left(|\textbf{B}| \textbf{B}^{-1}\right)_{ij}

		:return: cofactor matrix with shape, :math:`len(\text{systems}) \times n \times n`
		"""
		B_ij = np.zeros((self.z.shape[0], len(self.unique_mols), len(self.unique_mols), len(self.unique_mols), len(self.unique_mols)))
		for z_index in range(self.z.shape[0]):
			B_ij[z_index] = self._B_det[z_index] * self._B_inv[z_index]
		B_ij_tr = np.einsum('ijklm->ilmjk', B_ij)[:,:,:,:-1,:-1]
		return B_ij_tr

	@property
	def _rho_ij(self):
		'''product of rho's between two components'''
		_top_rho = np.zeros((self.z.shape[0], len(self.unique_mols), len(self.unique_mols)))
		for i, mol_1 in enumerate(self.unique_mols):
			rho_i = self.df_comp[f'rho_{mol_1}'].to_numpy()
			for j, mol_2 in enumerate(self.unique_mols):
				rho_j = self.df_comp[f'rho_{mol_2}'].to_numpy()
				_top_rho[:, i, j] = rho_i * rho_j
		return _top_rho

	def dmu_dN(self):
		r"""
		Calculate derivative of chemical potential with respect to number of molecules of type i

		.. math::
			\left(\frac{\partial \mu_i}{\partial N_j}\right)_{N, p, T} = \frac{k_bT}{\left<V\right> |\textbf{B}|}\left(\frac{\sum_{a=1}^N\sum_{b=1}^N \rho_a\rho_b\left|\textbf{B}^{ij}\textbf{B}^{ab}-\textbf{B}^{ai}\textbf{B}^{bj}\right|}{\sum_{a=1}^N\sum_{b=1}^N \rho_a\rho_b \textbf{B}^{ab}}\right)

		:return: np.array of shape, :math:`len(\text{systems}) \times n \times n`
		"""
		b_lower = np.zeros(self.z.shape[0]) # this matches!!
		for z_index in range(self.z.shape[0]):
			cofactors = self._B_det[z_index] * self._B_inv[z_index]
			b_lower[z_index] = np.einsum('ij,ij->', self._rho_ij[z_index], cofactors)

		# get system properties
		V = self.df_comp["box_vol"].to_numpy()
		n_tot = self.df_comp["n_tot"].to_numpy()

		# chemical potential derivative wrt molecule number
		dmu_dN_mat = np.full((self.z.shape[0], len(self.unique_mols), len(self.unique_mols)), fill_value=np.nan)
		for a in range(len(self.unique_mols)):
			for b in range(len(self.unique_mols)):
				b_upper = np.zeros(self.z.shape[0])
				for i, mol_1 in enumerate(self.unique_mols):
					for j, mol_2 in enumerate(self.unique_mols):
						b_upper += self._rho_ij[:,i,j] * np.linalg.det((self.cofactors_Bij()[:,a,b]*self.cofactors_Bij()[:,i,j] - self.cofactors_Bij()[:,i,a]*self.cofactors_Bij()[:,j,b]))
				b_frac = b_upper/b_lower
				dmu_dN_mat[:,a,b] = b_frac/(V*self._B_det)
		
		return dmu_dN_mat
		
	def dmu_dxs(self):
		r"""
		Calculate the chemical potential derivative with respect to mol fraction of each component.

		.. math::
			\left(\frac{\partial \mu_i}{\partial x_j}\right) = N \left(\frac{\partial \mu_i}{\partial N_j} - \frac{\partial \mu_i}{\partial N_n}\right)		
		
		:return: np.array of shape, :math:`len(\text{systems}) \times n-1 \times n-1`
		"""
		'''chemical potential derivatives'''
		# get chemical potential derivative wrt molecule number
		dmu_dN = self.dmu_dN()

		# get system properties
		V = self.df_comp["box_vol"].to_numpy()
		n_tot = self.df_comp["n_tot"].to_numpy()
		
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

		self._dmu_dxs = dmui_dxi
		return self._dmu_dxs
		
	def dlngamma_dxs(self):
		r"""
		Calculate the activity coefficient derivative with respect to mol fraction of molecule of type i

		.. math::
			\frac{\partial \gamma_i}{\partial x_i} = \frac{\partial \mu_i}{\partial x_i} - \frac{1}{x_i}
		
		:return: np.array of shape, :math:`len(\text{systems}) \times n`
		"""
		try:
			self._dmu_dxs
		except AttributeError:
			self.dmu_dxs()
		self._dlngamma_dxs = self._dmu_dxs - 1/self.z
		return self._dlngamma_dxs
	
	def _get_ref_state(self, mol):
		# get mol index
		i = self._mol_idx(mol=mol)
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

	def gammas(self):
		r"""
		Numerical integration of activity coefficient derivatives.
		
		.. math::
			\ln{\gamma_i}(x_i) =  \sum_{a_0}^{x_i} \frac{\Delta x}{2} \left[\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a_0} + \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a_1}\right]

		where :math:`a_0` is the reference state with :math:`a_0 = 1` for pure component and  :math:`a_0 = 0` for infinite dilution. 
		For :math:`a_0 = 1`,  :math:`a_1 = a_0 - \Delta x` and for :math:`a_0 = 0`, :math:`a_1 = a_0 + \Delta x`.

		:return: np.array of activity coefficients as a function of composition for each component
		"""

		try:
			self._dlngamma_dxs
		except AttributeError:
			self.dlngamma_dxs()

		dlny = self._dlngamma_dxs
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
			sorted_idxs = np.argsort(int_arr[:, 0])[::sort_idx]
			int_arr = int_arr[sorted_idxs]

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
			# get indices of filtered values
			filtered_sorted_idxs = np.delete(sorted_idxs, np.where(sorted_idxs==max(sorted_idxs))[0][0], axis=0)

			# revert back to original indices
			int_arr = int_arr[filtered_sorted_idxs]
			int_dlny_dx[:, i] = np.exp(int_arr[:, 2])

		# correct gammas for mean ionic activity coefficient if necessary
		gammas = self._correct_gammas(int_dlny_dx)
		# correct other properties that depend on geometric means
		self.z = self._correct_x(self.z) # correct the mol fractions 
		self.v = self._correct_x(self.v) # correct the vol fractions 
		self.unique_mols = self._correct_mols(self.unique_mols) # correct the unique mols 
		self.solute = self._correct_solute(self.solute) # reassign solute 

		self._gammas = gammas
		return self._gammas

	
	def _gamma_geom_mean(self, gammas, mol_1, mol_2):
		mol_1_idx = self._mol_idx(mol=mol_1)
		mol_2_idx = self._mol_idx(mol=mol_2)
		zi = self.mol_charge[mol_1_idx]
		zj = self.mol_charge[mol_2_idx]
		return (gammas[:,mol_1_idx]**(zi) * gammas[:,mol_2_idx]**(zj))**(1/(zi+zj))
	
	def _mol_idx(self, mol):
		return list(self.unique_mols).index(mol)
	
	def _correct_gammas(self, gammas):
		'''if geometric men activity coeff is present, adjust activity coeffs accordingly'''
		for i, (mol_1, mol_2) in enumerate(self.geom_mean_pairs):
			# get geometric mean of two components
			gamma_ij = self._gamma_geom_mean(gammas=gammas, mol_1=mol_1, mol_2=mol_2)
			# get molecule indices for removal
			mol_1_idx = self._mol_idx(mol=mol_1)
			mol_2_idx = self._mol_idx(mol=mol_2)
			# remove gammas of individual components from array
			gammas = np.delete(gammas, [mol_1_idx, mol_2_idx], axis=1)
			# add mean-ionic activity coefficient to array
			gammas = np.column_stack((gammas, gamma_ij))
		return gammas
	
	def _correct_x(self, x):
		'''if geometric mean exists also correct the mol fractions'''
		for i, (mol_1, mol_2) in enumerate(self.geom_mean_pairs):
			# get molecule indices for removal
			mol_1_idx = self._mol_idx(mol=mol_1)
			mol_2_idx = self._mol_idx(mol=mol_2)
			# get sum of two components
			sum_x = x[:,mol_1_idx] + x[:,mol_2_idx]
			# remove individual components from array
			x = np.delete(x, [mol_1_idx, mol_2_idx], axis=1)
			# add new component
			x = np.column_stack((x, sum_x))
		return x
	
	def _correct_mols(self, mols):
		for i, (mol_1, mol_2) in enumerate(self.geom_mean_pairs):
			mol_1_idx = self._mol_idx(mol=mol_1)
			mol_2_idx = self._mol_idx(mol=mol_2)
			new_mol_id = f"{mols[mol_1_idx]}-{mols[mol_2_idx]}"
			new_mol_name = f"{self.mol_name_dict[mol_1]}-{self.mol_name_dict[mol_2]}" 
			# remove individual names
			mols = np.delete(mols, [mol_1_idx, mol_2_idx])
			# add new molecule to mol_name_dict
			self.add_molname_to_dict(mol_id=new_mol_id, mol_name=new_mol_name)
		return mols

	def _correct_solute(self, solute):
		'''find solute'''
		# get unique mols from the geometric mean pairs
		unique_mols_gm = np.unique(self.geom_mean_pairs)
		if len(unique_mols_gm) > 0 and solute in unique_mols_gm:
			# if solute is in the geometric mean pairs, find the corresponding geometric mean molecule
			for mol_1 in unique_mols_gm:
				if mol_1 == solute:
					# find mols that contains solute name
					for mol_2 in self.unique_mols:
						if mol_1 in mol_2:
							return mol_2
		# if solute not in geometric mean pairs or no geometric mean pairs, return the original solute
		else:
			return solute
	
	def GE(self):
		r"""
		Excess Gibbs energy.
		
		.. math::
			G^E = RT\left(\sum_{i=1}^{n} x_i \ln{\gamma_i}\right)

		:return: 1D np.array for excess Gibbs energy as a function of composition
		"""
		try:
			self._gammas
		except AttributeError:
			self.gammas()
		self._GE = self.Rc * self.T_sim * (np.log(self._gammas) * self.z).sum(axis=1)
		return self._GE
	
	def G_id(self, x1_mat, x2_mat):
		return self.Rc * self.T_sim * (x1_mat * np.log(x2_mat)).sum(axis=1)
	
	@property
	def G_mix_xv(self):
		return self.GE() + self.G_id(self.z, self.v)

	@property
	def G_mix_xx(self):
		return self.GE() + self.G_id(self.z, self.z)
	
	@property
	def G_mix_vv(self):
		return self.GE() + self.G_id(self.v, self.v)

	def GM(self):
		r"""
		Gibbs mixing free energy.

		.. math::
			\Delta G_{mix} = RT\left(\sum_{i=1}^{n} x_i \ln{\left(\gamma_i x_i\right)}\right)
		
		:return: 1D np.array for Gibbs mixing free energy as a function of composition
		"""
		self._GM = self.G_mix_xv
		return self._GM

	@property
	def Hsim_pc(self):
		# first calculate H_pc for each component in system
		H_pc = {mol: 0 for mol in self.unique_mols}
		for i, mol in enumerate(self.unique_mols):
			# try and find directory
			sys = f"{mol}_{self.T_sim}"
			os.chdir(f"{self.pure_component_dir}/{sys}/")
			try:
				mols_present, total_num_mols, mol_nums_by_component = self._read_top(sys_parent_dir=self.pure_component_dir, sys=sys)
				# get npt edr files for system properties; volume, enthalpy.
				npt_edr_file = self._get_edr_file(sys=sys)
				# get simulation enthalpy
				if os.path.exists('enthalpy_npt.xvg') == False:
					os.system(f"echo enthalpy | gmx energy -f {npt_edr_file} -o enthalpy_npt.xvg")
				time, H = np.loadtxt('enthalpy_npt.xvg', comments=["#", "@"], unpack=True)
				H_pc[mol] = self._get_time_average(time, H)/total_num_mols		
			except:
				# if file/path does not exist just use nan values
				H_pc[mol] = np.nan
		return H_pc

	@property
	def md_molar_vol(self):
		# get molar volume of pure components
		vol = np.zeros(self.unique_mols.size)
		for i, mol in enumerate(self.unique_mols):
			# try and find directory
			sys = f"{mol}_{self.T_sim}"
			os.chdir(f"{self.pure_component_dir}/{sys}/")
			mols_present, total_num_mols, mol_nums_by_component = self._read_top(sys_parent_dir=self.pure_component_dir, sys=sys)
			# get npt edr files for system properties; volume, enthalpy.
			npt_edr_file = self._get_edr_file(sys=sys)
			# get simulation density
			if os.path.exists('density_npt.xvg') == False:
				os.system(f"echo density | gmx energy -f {npt_edr_file} -o density_npt.xvg")
			time, rho = np.loadtxt('density_npt.xvg', comments=["#","@"], unpack=True)
			density = self._get_time_average(time, rho) / 1000 # g/mL		
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

	def Hmix(self):
		r"""
		Mixing enthalpy from molecular simulation.

		.. math::
			\Delta H_{mix} = H - \sum_{i=1}^{n} x_i H_{i}^{pc} \label{eqn:Hmix}

		where :math:`H` is the enthalpy for a simulation at a given composition and :math:`H_i^{pc}` is the pure component enthalpy for molecule of type i.
		
		:return: 1D np.array for enthalpy as a function of composition
		"""

		self._Hmix = self.Hsim - self.H_id_mix
		return self._Hmix
	
	def SE(self):
		r"""
		Excess entropy from Gibbs excess property relations.
		
		.. math::
			S^E = \frac{\Delta H_{mix} - G^E}{T}

		:return: 1D np.array for excess entropy as a function of composition
		"""
		try:
			self._Hmix
		except AttributeError:
			self.Hmix()
		self._SE = (self._Hmix - self.GE()) / self.T_sim
		return self._SE
	
	def SM(self):
		try:
			self._Hmix
		except AttributeError:
			self.Hmix()
		self._SM = (self._Hmix - self.GM()) / self.T_sim
		return self._SM

	@property
	def nTdSmix(self):
		try:
			self._Hmix
		except AttributeError:
			self.Hmix()
		return self.G_mix_xv - self._Hmix

	@property
	def nrtl_taus(self):
		return self._fit_NRTL_IP()
	
	def _fit_NRTL_IP(self):
		r"""
		Fit NRTL interaction parameters (taus) to Gibbs mixing free energy.

		.. math::
			\Delta G_{mix}^{NRTL} = -RT\left(x_1 x_2 \left(x_1 \ln{\left(x_1\right)} + x_2 \ln{\left(x_2\right)} + \frac{\tau_{21} G_{21}}{x_1 + x_2 G_{21}} + \frac{\tau_{12} G_{12}}{x_2 + x_1 G_{12}} \right)\right)

		.. math::
			G_{12} = \exp{\left(\frac{-\alpha \tau_{12}}{RT}\right)}

		.. math::
			G_{21} = \exp{\left(\frac{-\alpha \tau_{21}}{RT}\right)}

		:return: dictionary of nrtl regressed taus
		"""
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
	

	def _fit_FH_chi(self):
		r"""
		Fit Flory-Huggins chi parameter to Gibbs mixing free energy.

		.. math::
			\Delta G_{mix}^{FH} = RT\left(\phi_1 \frac{\phi_1}{N_1} + \phi_2 \frac{\phi_2}{N_2} \right) + \chi x_1 x_2

		:return: chi interaction parameter
		"""
		if len(self.unique_mols) != 2:
			# check that system is binary, else don't run
			return
			
		phi = self.v[:,self.solute_loc]
		self.fh_phi = phi

		N0 = self.molar_vol / self.molar_vol.max() # normalize the molar volumes to get N0 for Flory-Huggins

		def fh_GM(x, chi, Tx):
			return 8.314E-3 * Tx * (x * np.log(x)/N0[0] + (1-x) * np.log(1-x)/N0[1]) + chi*x*(1-x)

		GM_fit_Tsim = partial(fh_GM, Tx=self.T_sim)
		
		fit, pcov = curve_fit(GM_fit_Tsim, xdata=phi, ydata=self.G_mix_xx)
		chi = fit[0]
		
		self.fh_chi = chi
		self.fh_Gmix = fh_GM(phi, chi, self.T_sim)

		with open(f'{self.kbi_method_dir}FH_chi_{self.kbi_method.lower()}.txt', 'w') as f:
			f.write(f'{self.fh_chi}\n')
	
		return self.fh_chi

	@property
	def uniquac_du(self):
		return self._fit_UNIQUAC_IP()

	def _fit_UNIQUAC_IP(self):
		r"""
		Fit UNIQUAC interaction parameters to mixing enthalpy.

		.. math::
			\Delta H_{mix}^{UNIQUAC} = -RT \left(\sum_i q_i x_i \ln{\left(\sum_j \theta_j \tau_{ji} \right)}\right)

		.. math::
			\tau_{ji} = \exp{\left(\frac{-\Delta u_{ji}}{RT}\right)}	
		"""
		try:
			self._Hmix
		except AttributeError:
			self.Hmix()
		self.r, self.q = UNIQUAC_RQ(self.smiles)
		du = fit_du_to_Hmix(z=self.z, Hmix=self._Hmix, T=self.T_sim, smiles=self.smiles)
		np.savetxt(f"{self.kbi_method_dir}UNIQUAC_du_{self.kbi_method.lower()}.txt", du, delimiter=",") 
		return du

	@property
	def z_plot(self):
		# z-matrix for thermoydnamic model evaluations
		num_pts = {2:10, 3:7, 4:6, 5:5, 6:4}
		point_disc = PointDisc(num_comp=self.z.shape[1], recursion_steps=num_pts[self.z.shape[1]], load=True, store=False)
		z_arr = point_disc.points_mfr[1:-1,:]
		return z_arr 

	@property
	def uniquac_Hmix(self):
		try:
			self.uniquac_du
		except:
			self._fit_UNIQUAC_IP()

		uniq_model = UNIQUAC(z=self.z_plot, smiles=self.smiles, IP=self.uniquac_du)
		return uniq_model.GE_res(self.T_sim)

	@property
	def uniquac_Smix(self):
		try:
			self.uniquac_du
		except:
			self._fit_UNIQUAC_IP()

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
		try:
			self._Hmix
		except AttributeError:
			self.Hmix()
		quar_model = QuarticModel(z_data=self.z, z=self.z_plot, Hmix=self._Hmix, Sex=self.SE(), molar_vol=self.molar_vol)
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


	

						







						



