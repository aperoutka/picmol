import numpy as np
import os
import sys
from pathlib import Path

from .unifac_subgroups.unifac_subgroup_parameters import UFIP, UFSG, UFILIP, UFILSG
from .unifac_subgroups.fragmentation import Groups

# core functions to be used in unifac class for mixtures
def unifac_R(subgroup_dict: dict, subgroup_data):
	'''get UNIFAC R "volume" parameter
	r: float = molecule parameter'''
	return sum([occurence * subgroup_data[group].R for group, occurence in subgroup_dict.items()])


def unifac_Q(subgroup_dict: dict, subgroup_data):
	'''get UNIFAC Q "surface area" parameter
	q: float = molecule parameter'''
	return sum([occurence * subgroup_data[group].Q for group, occurence in subgroup_dict.items()])


def unifac_psi(T, subgroup1, subgroup2, subgroup_data, interaction_data, version):
	"""get interaction parameters for a given subgroup combination"""
	main1 = subgroup_data[subgroup1].main_group_id
	main2 = subgroup_data[subgroup2].main_group_id
	try: # for cross-terms
		a = interaction_data[main1][main2]
		return np.exp(-a/T)
	except: # for same group interaction
		return 1. 
		

def unifac_psis_matrix(T, unique_subgroups, subgroup_data, interaction_data, version):
	"""get interaction parameters for a given subgroup combination"""

	# map subgorup IDs to main group IDs
	subgroup_ids = np.unique(unique_subgroups)
	subgroup_to_main = {s: subgroup_data[s].main_group_id for s in subgroup_ids}

	# create index mapping for main group IDs
	main_group_ids = np.array([subgroup_to_main[s] for s in subgroup_ids])

	# Create 2D grids of main group IDs for all pairs
	main1_grid, main2_grid = np.meshgrid(main_group_ids, main_group_ids, indexing='ij')

	# Initialize the psi matrix with ones
	psi_matrix = np.ones_like(main1_grid, dtype=float)

	# Identify same group interactions
	same_group = main1_grid == main2_grid

	a_matrix = np.zeros_like(main1_grid, dtype=float)

	# Create masks for unique pairs of main groups
	unique_pairs = np.unique(
			np.stack((main1_grid[~same_group], main2_grid[~same_group]), axis=1), axis=0
	)

	# Assign interaction parameters to the matrices
	for m1, m2 in unique_pairs:
			mask = (main1_grid == m1) & (main2_grid == m2)
			try:
					a_val = interaction_data[m1][m2]
					a_matrix[mask] = a_val
			except KeyError:
					# Missing interaction parameters; psi remains as 1.0
					pass

	# Compute psi values where main groups are different
	psi_matrix[~same_group] = np.exp(-a_matrix[~same_group] / T)

	# psi_matrix already has ones on the diagonal for same group interactions

	return psi_matrix


class UNIFAC:

	'''using uniquac, calculate the mixing free energy and derivatives for a matrix'''

	def __init__(self, T: float, smiles: list, z = None):
		"""
		parameters:
		z: mol fractions (type: array)
		T: temperature [K]
		subgroup_dict: unifac subgroups (type: dict) by group_id: occurance
		interaction_parameters: (type: dict) -- UFIP, UFILIP
		version: unifac-il or unifac
		"""

		self.zc = 10
		self.T = T
		self.Rc = 8.314E-3
		if z is not None:
			self.z = z
		else:
			self.z = np.load(Path(__file__).parent / "molfr_matrix" / f"{len(smiles)}_molfr_pts.npy")
		
		# get sugroups in each molecule
		subgroup_nums = []
		for smile in smiles:
			group_obj = Groups(smile, "smiles")
			if '.' in smile:
				subgroup_nums.append(group_obj.unifac_IL.to_num)
			else:
				subgroup_nums.append(group_obj.unifac.to_num)
		self.subgroups = subgroup_nums

		# get version baased on smiles
		version = ['il' for smile in smiles if '.' in smile]

		if len(version) > 0:
			version = 'unifac-il'
		else:
			version = 'unifac'

		# get a numeric key for unifac version
		# version_key: version_value
		version_dict = {'unifac': 0,
										'unifac-il': 1}
		self.version = version_dict[version.lower()]

		# version: interaction parameters dictionary
		ip_dict = {0: UFIP, 1: UFILIP}
		self.interaction_data = ip_dict[self.version]

		# subgroup data
		sg_dict = {0: UFSG, 1: UFILSG}
		self.subgroup_data = sg_dict[self.version]

	def unique_groups(self):
		'''unique subgroups present'''
		all_subgroups = []
		for mol in self.subgroups:
			all_subgroups.extend(list(mol.keys()))
		self._unique_groups = np.unique(all_subgroups)
		return self._unique_groups

	@property
	def N_groups(self):
		try:
			self._unique_groups
		except AttributeError:
			self.unique_groups()
		return len(self._unique_groups)
	
	@property
	def N(self):
		return len(self.subgroups)


	def occurrance_matrix(self):
		'''create a matrix of subgroup occurrance for each molecule'''
		try:
			self._unique_groups
		except:
			self.unique_groups()
		occurrance_matrix = np.zeros((self.N_groups, self.N))
		for i, mol in enumerate(self.subgroups):
			subgroup_arr = np.array(list(mol.keys()))
			occurance_arr = np.array(list(mol.values()))
			for g, group in enumerate(subgroup_arr):
				occurrance_matrix[self._unique_groups == group, i] = occurance_arr[g]
		self._occurrance_matrix = occurrance_matrix
		return self._occurrance_matrix


	def subgroup_matrix(self):
		'''create a matrix of subgroups for each molecule'''
		try:
			self._unique_groups
		except:
			self.unique_groups()
		subgroup_matrix = np.zeros((self.N_groups, self.N))
		for i, mol in enumerate(self.subgroups):
			subgroup_arr = np.array(list(mol.keys()))
			occurance_arr = np.array(list(mol.values()))
			for g, group in enumerate(subgroup_arr):
				subgroup_matrix[self._unique_groups == group, i] = subgroup_arr[g]
		self._subgroup_matrix = subgroup_matrix
		return self._subgroup_matrix
	

	def Q_matrix(self):
		'''create a matrix of Q values'''
		try:
			self._subgroup_matrix
		except AttributeError:
			self.subgroup_matrix()
		Q_matrix = np.zeros(np.shape(self._subgroup_matrix))
		for row in range(np.shape(self._subgroup_matrix)[0]):
			for col in range(np.shape(self._subgroup_matrix)[1]):
				if self._subgroup_matrix[row,col] != 0.:
					Q_matrix[row,col] = self.subgroup_data[self._subgroup_matrix[row,col]].Q
		self._Q_matrix = Q_matrix
		return self._Q_matrix
	

	def R(self):
		'''unifac volume parameter'''
		self._r = np.array([unifac_R(mol_subgroups, self.subgroup_data) for mol_subgroups in self.subgroups])
		return self._r


	def Q(self):
		'''unifac area parameter'''
		self._q = np.array([unifac_Q(mol_subgroups, self.subgroup_data) for mol_subgroups in self.subgroups])
		return self._q
	
	
	def psis(self):
		try:
			self._unique_groups
		except AttributeError: 
			self.unique_groups()
		self._psis = unifac_psis_matrix(T=self.T, unique_subgroups=self._unique_groups, subgroup_data=self.subgroup_data, interaction_data=self.interaction_data, version=self.version)
		return self._psis


	def group_counts(self):
		'''number of unique subgroups present'''
		try:
			self._unique_groups
		except AttributeError:
			self.unique_groups()
		group_counts = {}
		for i in range(len(self._unique_groups)):
			for group, occurrance in self.subgroups[i].items():
				group_counts[group] += occurrance
		self._group_counts = np.array(list(group_counts.values()))
		return self._group_counts
	

	def weighted_number(self):
		'''weighted X, by occurrance; returns a 3D matrix'''
		try:
			self._occurrance_matrix
		except AttributeError:
			self.occurrance_matrix()	

		self._weighted_number = self.z[:,np.newaxis,:] * self._occurrance_matrix[np.newaxis,:,:]
		return self._weighted_number


	def group_X(self):
		'''mol fraction of groups in a mixture'''
		try:
			self._weighted_number
		except AttributeError:
			self.weighted_number()	
		try:
			self._unique_groups
		except AttributeError:
			self.unique_groups()
		# get the sum of each array in weighted_number
		sum_weights = np.sum(self._weighted_number, axis=(2,1))
		frac_weights = self._weighted_number / sum_weights[:,np.newaxis, np.newaxis]
		# create a matrix of unique groups x number of compositions
		group_X = np.zeros((np.shape(self.z)[0], len(self._unique_groups)))
		for i in range(np.shape(frac_weights)[0]):
			for g, group in enumerate(self._unique_groups):
				group_X[i,g] = np.sum(frac_weights[i][self._unique_groups == group])
		self._group_X = group_X
		return self._group_X
	

	def group_Q(self):
		'''Q for unique groups'''
		try:
			self._unique_groups
		except AttributeError:
			self.unique_groups()	
		group_Q = np.zeros(len(self._unique_groups))
		for s, subgroup in enumerate(self._unique_groups):
			group_Q[s] = self.subgroup_data[subgroup].Q
		self._group_Q = group_Q
		return self._group_Q


	def thetas(self):
		'''
		Area term for each molecule in mixture

		.. math::
			\theta_i = \Frac{x_i * q_i}{\sum_{j=1}^{n} x_j * q_j}
		'''
		try:
			self._Ais
		except AttributeError:
			self.Ais()
		self._thetas = self._Ais * self.z
		return self._thetas
	

	def Thetas(self):
		'''
		Area term for each group in mixture

		.. math::
			\Theta_i = \Frac{X_i * Q_i}{\sum_{j=1}^{n} X_j * Q_j}
		'''
		try:
			self._group_Q
		except AttributeError:
			self.group_Q()
		try: 
			self._group_X
		except AttributeError:
			self.group_X()
		# instead of going by each molecule, this requires going by each subgroup composition
		Thetas = (self._group_X * self._group_Q).T / (self._group_X @ self._group_Q) 
		self._Thetas = Thetas.T # change to composition x groups
		return self._Thetas

	def rbar(self):
		try:
			self._r
		except AttributeError:
			self.R()
		self._rbar = self.z @ self._r # takes the dot product
		return self._rbar
	
	def Vis(self):
		'''
		Volume term for each molecule in mixture, without molar fraction contribution of i
		
		.. math::
			\Vis_i = \Frac{r_i}{\sum_{j=1}^{n} x_j * r_j}
		'''
		try:
			self._rbar
		except AttributeError:
			self.rbar()
		self._Vis = self._r / self._rbar[:,np.newaxis] # divides each row in r by column in rbar
		return self._Vis 
	
	def qbar(self):
		try:
			self._q
		except AttributeError:
			self.Q()
		self._qbar = self.z @ self._q # takes the dot product
		return self._qbar

	def Ais(self):
		'''
		Area term for each molecule in mixture, without molar fraction contribution of i
		
		.. math::
			\Ais_i = \Frac{q_i}{\sum_{j=1}^{n} x_j * q_j}
		'''
		try:
			self._qbar
		except AttributeError:
			self.qbar()
		self._Ais = self._q / self._qbar[:,np.newaxis] # divides each row in q by column in qbar
		return self._Ais

	def phis(self):
		'''
		Volume term for each molecule in mixture

		.. math::
			\phis_i = \Frac{x_i * q_i}{\sum_{j=1}^{n} x_j * q_j}
		'''
		try:
			self._Vis
		except AttributeError:
			self.Vis()
		self._phis = self._Vis * self.z
		return self._phis
		
	
	def gammas(self):
		"""
		total activity coefficients
		
		..math::
			\ln \gamma_i = \ln \gamma_i^c + \ln \gamma_i^r
		"""
		try:
			self._lngammas_c
		except AttributeError:
			self.lngammas_c()
		try:
			self._lngammas_r
		except AttributeError:
			self.lngammas_r()
		self._gammas = np.exp(self._lngammas_c + self._lngammas_r)
		return self._gammas


	def lngammas_c(self):
		"""
		get combinatorial activity coefficients

		..math::
			\gamma_i^c = \ln\frac{\phi_i}{x_i} + (self.z/2) * q_i * \ln\frac{\theta_i}{\phi_i} + \ell_i - \frac{\phi_i}{x_i} * \sum_{j=1} x_j * \ell_j
			\ell_i = (self.z/2) * (r_i - q_i) - (r_i - 1)
			z = 10
		"""
		try:
			self._q
		except AttributeError:
			self.Q()
		try:
			self._r
		except AttributeError:
			self.R()
		try:
			self._thetas
		except AttributeError:
			self.thetas()
		try:
			self._phis
		except AttributeError: 
			self.phis()
		self._ell = (self.zc/2) * (self._r - self._q) - (self._r - 1)
		self._lngammas_c = np.log(self._phis/self.z) + (self.zc/2) * self._q * np.log(self._thetas/self._phis) + self._ell - (self._phis/self.z) * (self.z @ self._ell)[:,np.newaxis]
		return self._lngammas_c
	

	def lngammas_r(self):
		"""
		get residual activity coefficients

		..math::
			\gamma_i^r = \sum_{k=1} \nu_k^(i) * [\ln{\Gamma_k} - \ln{\Gamma_k^(i)}]
		"""
		try:
			self._lnGammas_subgroups
		except AttributeError:
			self.lnGammas_subgroups()
		try:
			self._lnGammas_subgroups_pure
		except AttributeError:
			self.lnGammas_subgroups_pure()
		
		self._lngammas_r = np.sum(self._occurrance_matrix * (self._lnGammas_subgroups[:,np.newaxis] - self._lnGammas_subgroups_pure), axis=2)[:,0,:]
		return self._lngammas_r


	def X_pure(self):
		"""group fractions for each molecule"""
		try:
			self._occurrance_matrix
		except AttributeError:
			self.occurrance_matrix()
		self._X_pure = self._occurrance_matrix / np.sum(self._occurrance_matrix, axis=0)
		return self._X_pure
	

	def Thetas_pure(self):
		"""area group fractions for each molecule"""
		try:
			self._Q_matrix
		except AttributeError:
			self.Q_matrix()
		try:
			self._X_pure
		except AttributeError:
			self.X_pure()
		self._Thetas_pure = (self._X_pure * self._Q_matrix) / np.sum(self._X_pure * self._Q_matrix, axis=0)
		return self._Thetas_pure
	
	
	def lnGammas_subgroups_pure(self):
		"""
		residual activity coefficient of group k in a ref. solution containing only molecules of type i

		..math::
			\Gamma_k = Q_k * [1 - \ln{\sum_{m=1} \Theta_m * \Psi_{mk}} - \sum_{m=1} \frac{\Theta_m * \Psi_{km}}{\sum_{n=1} \Theta_n * \Psi_{nm}}
		"""
		try:
			self._Thetas_pure
		except AttributeError:
			self.Thetas_pure()
		try:
			self._Q_matrix
		except AttributeError:
			self.Q_matrix()
		try:
			self._psis
		except AttributeError: 
			self.psis()

		thetas_psis_12 = (self._Thetas_pure[:,np.newaxis,:] * self._psis[:,:,np.newaxis]).sum(axis=0)
		thetas_psis_21 = self._Thetas_pure[np.newaxis,:,:] * self._psis[:,:,np.newaxis]
		sum_thetas_psis_12_thetas_psis_21 = (thetas_psis_21/thetas_psis_12).sum(axis=1)
		self._lnGammas_subgroups_pure = self._Q_matrix * (1 - np.log(thetas_psis_12) - sum_thetas_psis_12_thetas_psis_21)
		return self._lnGammas_subgroups_pure


	def lnGammas_subgroups(self):
		"""
		residual activity coefficient of group k in mixture

		..math::
			\Gamma_k = Q_k * [1 - \ln{\sum_{m=1} \Theta_m * \Psi_{mk}} - \sum_{m=1} \frac{\Theta_m * \Psi_{km}}{\sum_{n=1} \Theta_n * \Psi_{nm}}
		"""
		try:
			self._Thetas
		except AttributeError:
			self.Thetas()
		try:
			self._psis
		except AttributeError: 
			self.psis()
		try:
			self._subgroup_matrix
		except:
			self.subgroup_matrix()

		Theta_psi = self._Thetas @ self._psis # shape: (S, G)

		# expand dimensions for broadcasting
		Theta_m = self._Thetas[:,np.newaxis,:] # shape: (S, 1, G)
		psi_km = self._psis[np.newaxis,:,:] # shape: (1, G, G)
		Theta_psi_nm = Theta_psi[:,np.newaxis,:] # shape: (S, 1, G)

		# compute theta * psi / (sum{theta*psi})
		Theta_psi_km = Theta_m * psi_km # shape: (S, G, G)
		sum_Theta_psi_km_Theta_psi_nm = (Theta_psi_km / Theta_psi_nm).sum(axis=2) # shape: (S, G)

		# compute the residual activity coef. for each group in each component
		subgroup_lnGammas_subgroups = self._group_Q[np.newaxis,:] * (1 - np.log(Theta_psi) - sum_Theta_psi_km_Theta_psi_nm) # shape: (S, G)

		# convert to appropriate dimensions
		subgroup_lnGammas_subgroups_matrix = np.zeros(np.shape(self._weighted_number))
		for i in range(np.shape(self._weighted_number)[0]):
			for g, group in enumerate(self._unique_groups):
				subgroup_lnGammas_subgroups_matrix[i][self._subgroup_matrix == group] = subgroup_lnGammas_subgroups[i,g]

		self._lnGammas_subgroups = subgroup_lnGammas_subgroups_matrix
		return self._lnGammas_subgroups
	

	def GE(self):
		"""
		Gibbs excess energy

		..math:
			GE = R * T *  \sum_{i=1} x_i * log(gamma_i)
		"""
		try:
			self._gammas
		except AttributeError: 
			self.gammas()
		self._GE = (self.Rc * self.T * np.sum(self.z * np.log(self._gammas), axis=1))
		return self._GE
	

	def GM(self):
		"""
		Gibbs mixing energy

		..math:
			GM = R * T *  \sum_{i=1} x_i * log(gamma_i * x_i)
		"""
		try:
			self._gammas
		except AttributeError: 
			self.gammas()
		self._GM = (self.Rc * self.T * np.sum(self.z * np.log(self._gammas * self.z), axis=1))
		return self._GM
	
	
	def dlngammas_c_dxs(self):

		try:
			self._dVis_dxs
		except AttributeError: 
			self.dVis_dxs()
		try:
			self._dAis_dxs
		except AttributeError: 
			self.dAis_dxs()
		try:
			self._Vis
		except AttributeError: 
			self.Vis()
		try:
			self._Ais
		except AttributeError: 
			self.Ais()
	
		Vis = self._Vis[:,np.newaxis,:]
		Ais = self._Ais[:,np.newaxis,:]
		dlngammas_c_dxs = -5*self._q* ((self._dVis_dxs/Vis)-(Vis*self._dAis_dxs/Ais**2)*(Ais/Vis) - (self._dVis_dxs/Ais) + (Vis*self._dAis_dxs/Ais**2)) - self._dVis_dxs + (self._dVis_dxs/Vis)
		self._dlngammas_c_dxs = dlngammas_c_dxs
		return self._dlngammas_c_dxs


	def dlngammas_r_dxs(self):

		try:
			self._occurrance_matrix
		except AttributeError: 
			self.occurrance_matrix()
		try:
			self._dlnGammas_subgroups_dxs 
		except AttributeError: 
			self.dlnGammas_subgroups_dxs()

		dlngammas_r_dxs = self._occurrance_matrix.T @ self._dlnGammas_subgroups_dxs
		dlngammas_r_dxs = dlngammas_r_dxs.reshape(np.shape(self.z)[0], self.N, self.N)
		self._dlngammas_r_dxs = dlngammas_r_dxs
		return self._dlngammas_r_dxs


	def dAis_dxs(self):

		try:
			self._qbar
		except AttributeError: 
			self.Ais()
		
		self._dAis_dxs = (-np.outer(self._q, self._q)) / self._qbar[:,np.newaxis,np.newaxis]**2
		return self._dAis_dxs


	def dVis_dxs(self):

		try:
			self._rbar
		except AttributeError: 
			self.rbar()
		
		self._dVis_dxs = (-np.outer(self._r, self._r)) / self._rbar[:,np.newaxis,np.newaxis]**2
		return self._dVis_dxs
	

	def F(self):
		try:
			self._weighted_number
		except AttributeError: 
			self.weighted_number()
		self._F = 1/(self._weighted_number.sum(axis=1)).sum(axis=1) 
		return self._F
	

	def G(self):
		try:
			self._group_X
		except AttributeError:
			self.group_X()
		try:
			self._group_Q
		except AttributeError: 
			self.group_Q()
		self._G = 1/(self._group_X * self._group_Q).sum(axis=1)
		return self._G
	

	def sum_occurance_matrix(self):
		# ie, VS in thermo pkg
		try:
			self._occurrance_matrix
		except AttributeError: 
			self.occurrance_matrix()
		self._sum_occurance_matrix = self._occurrance_matrix.sum(axis=0)
		return self._sum_occurance_matrix
	

	def sum_weighted_number(self):
		# ie., VSXS in thermo pkg
		try:
			self._weighted_number
		except AttributeError: 
			self.weighted_number()
		self._sum_weighted_number = self._weighted_number.sum(axis=2)
		return self._sum_weighted_number



	def dThetas_dxs(self):

		try:
			self._F
		except AttributeError:
			self.F()
		try:
			self._G
		except AttributeError:
			self.G()
		try:
			self._sum_occurance_matrix
		except AttributeError: 
			self.sum_occurance_matrix()
		try:
			self._sum_weighted_number
		except AttributeError: 
			self.sum_weighted_number()

		lenF = self._F.shape[0] # Number of F and G values

		# Initialize output arrays 
		dThetas_dxs = np.zeros((lenF, self.N_groups, self.N))
		vec0 = np.zeros((lenF, self.N))

		# Compute tot0: F multiplied by the sum over N_groups of sum_weighted_number * self._group_Q
		tot0 = self._F * np.sum(self._sum_weighted_number * self._group_Q[np.newaxis, :], axis=1)

		# Compute tot1: Negative sum over N_groups of self._group_Q * vs
		tot1 = -np.sum(self._group_Q[:, np.newaxis] * self._occurrance_matrix, axis=0)

		# Compute vec0: Vector of size (lenF, N)
		# Expand dimensions for broadcasting
		tot0_expanded = tot0[:, np.newaxis]
		F_expanded = self._F[:, np.newaxis]
		G_expanded = self._G[:, np.newaxis]
		sum_occurance_matrix_expanded = self._sum_occurance_matrix[np.newaxis, :]
		tot1_expanded = tot1[np.newaxis, :]

		# Compute intermediate values
		temp = tot0_expanded * sum_occurance_matrix_expanded + tot1_expanded
		vec0 = F_expanded * (G_expanded * temp - sum_occurance_matrix_expanded)

		# Compute ci: (F * G) outer product with self._group_Q
		ci = (self._F * self._G)[:, np.newaxis] * self._group_Q[np.newaxis, :]

		# Compute the outer product of sum_weighted_number and vec0
		outer = self._sum_weighted_number[:, :, np.newaxis] * vec0[:, np.newaxis, :]

		# Broadcast vs to match dimensions
		vs_expanded = self._occurrance_matrix[np.newaxis, :, :]

		# Compute dThetas_dxs
		dThetas_dxs = ci[:, :, np.newaxis] * (outer + vs_expanded)

		self._dThetas_dxs = dThetas_dxs
		return self._dThetas_dxs


	def Ws(self):
		try:
			self._psis
		except AttributeError:
			self.psis()
		try:
			self._dThetas_dxs
		except AttributeError:
			self.dThetas_dxs()
		self._Ws =  (self._psis[np.newaxis,:,:,np.newaxis] * self._dThetas_dxs[:,:,np.newaxis,:]).sum(axis=1)
		return self._Ws
	

	def Theta_Psi_sum_invs(self):
		try:
			self._Thetas
		except AttributeError: 
			self.Thetas()
		try:
			self._psis
		except AttributeError:
			self.psis()
		self._Theta_Psi_sum_invs = 1/(self._Thetas @ self._psis)
		return self._Theta_Psi_sum_invs
	

	def dlnGammas_subgroups_dxs(self):
		try:
			self._Ws
		except AttributeError:
			self.Ws()
		try:
			self._Theta_Psi_sum_invs
		except AttributeError:
			self.Theta_Psi_sum_invs()
		
		### Step 1: Compute the First Term
		# First_term[l, k, i] = -Ws[l, k, i] * Theta_Psi_sum_invs[l, k]
		First_term = -self._Ws * self._Theta_Psi_sum_invs[:, :, np.newaxis]  # Shape: (L, N_groups, N)

		### Step 2: Compute Intermediate Values
		# Compute Ws_TPT_inv_Thetas[l, m, i] = Ws[l, m, i] * Theta_Psi_sum_invs[l, m] * Thetas[l, m]
		Ws_TPT_inv_Thetas =self._Ws * self._Theta_Psi_sum_invs[:, :, np.newaxis] * self._Thetas[:, :, np.newaxis]  # Shape: (L, N_groups, N)

		# Compute Delta_dThetas_dxs[l, m, i] = dThetas_dxs[m, i] - Ws_TPT_inv_Thetas[l, m, i]
		Delta_dThetas_dxs = self._dThetas_dxs[np.newaxis, :, :] - Ws_TPT_inv_Thetas  # Shape: (L, N_groups, N)

		### Step 3: Compute the Second Term using Batch Matrix Multiplication
		# Compute A[l, k, m] = psis[k, m] * Theta_Psi_sum_invs[l, m]
		psis_sum_Theta_psis_inv = self._psis[np.newaxis, :, :] * self._Theta_Psi_sum_invs[:, np.newaxis, :]  # Shape: (L, N_groups, N_groups)

		# Compute Second_term[l, k, i] = sum over m [A[l, k, m] * Delta_dThetas_dxs[l, m, i]]
		Second_term = np.matmul(psis_sum_Theta_psis_inv, Delta_dThetas_dxs)  # Shape: (L, N_groups, N)

		### Step 4: Compute Total and Final Result
		Total = First_term - Second_term  # Shape: (L, N_groups, N)

		# Multiply by self._group_Q to get the final result
		self._dlnGammas_subgroups_dxs = Total * self._group_Q[np.newaxis, :, np.newaxis]  # Shape: (L, N_groups, N)
		return self._dlnGammas_subgroups_dxs


	def dGE_dxs(self):
		
		try:
			self._dlngammas_c_dxs
		except AttributeError: 
			self.dlngammas_c_dxs()
		try:
			self._lngammas_r_dxs
		except AttributeError: 
			self.dlngammas_r_dxs()
		try:
			self._lngammas_c
		except AttributeError:
			self.lngammas_c()
		try:
			self._lngammas_r
		except AttributeError:
			self.lngammas_r()

		lngammas = self._lngammas_c + self._lngammas_r
		dlngammas = np.sum(self.z[:,np.newaxis,:] * (self._dlngammas_c_dxs + self._dlngammas_r_dxs), axis=2)

		self._dGE_dxs = self.Rc * self.T * (lngammas + dlngammas)

		return self._dGE_dxs


	def mu(self):
		
		try:
			self._dlngammas_c_dxs
		except AttributeError: 
			self.dlngammas_c_dxs()
		try:
			self._lngammas_r_dxs
		except AttributeError: 
			self.dlngammas_r_dxs()
		try:
			self._lngammas_c
		except AttributeError:
			self.lngammas_c()
		try:
			self._lngammas_r
		except AttributeError:
			self.lngammas_r()

		lngammas = self._lngammas_c + self._lngammas_r
		dlngammas = np.sum(self.z[:,np.newaxis,:] * (self._dlngammas_c_dxs + self._dlngammas_r_dxs), axis=2)

		self._mu = self.Rc * self.T * (lngammas + dlngammas + np.log(self.z) + 1)

		return self._mu
	

	def dGM_dxs(self):
		'''calculates the first derivative of mixing free energy'''
		try:
			self._mu 
		except:
			self.mu()
		
		dGM_dxs = np.zeros((self.z.shape[0], self.z.shape[1]-1))
		for i in range(self.z.shape[1]-1):
			dGM_dxs[:,i] = self._mu[:,i] - self._mu[:,self.z.shape[1]-1]
		self._dGM_dxs = dGM_dxs
		return self._dGM_dxs
	
	
	def d2lngammas_c_dxixjs(self):
		try:
			self._Vis
		except AttributeError:
			self.Vis()
		try:
			self._Ais
		except AttributeError:
			self.Ais()
		try:
			self._dVis_dxs
		except AttributeError: 
			self.dVis_dxs()
		try:
			self._dAis_dxs
		except AttributeError: 
			self.dAis_dxs()
		try:
			self._d2Vis_dxixjs
		except AttributeError: 
			self.d2Vis_dxixjs()
		try:
			self._d2Ais_dxixjs
		except AttributeError: 
			self.d2Ais_dxixjs()

		# Precompute repeated terms
		Vi_inv2 = 1.0 / (self._Vis ** 2)  # Shape: (3, N)
		x1 = 1.0 / self._Ais  # Shape: (3, N)
		x4 = x1 ** 2  # Shape: (3, N)
		Ai_inv3 = x1 ** 3  # Shape: (3, N)
		x5 = self._Vis * x4  # Shape: (3, N)
		x15 = 1.0 / self._Vis  # Shape: (3, N)
		Vi_inv2 = x15 ** 2  # Shape: (3, N)

		# Expand dimensions for broadcasting
		Vi_expanded = self._Vis[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
		qi_expanded = self._q[np.newaxis, :, np.newaxis, np.newaxis]  # Shape: (1, N, 1, 1)
		Vi_inv2_expanded = Vi_inv2[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
		x1_expanded = x1[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
		x4_expanded = x4[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
		Ai_inv3_expanded = Ai_inv3[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
		x5_expanded = x5[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
		x15_expanded = x15[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)
		Vi_inv2_expanded = Vi_inv2[:, :, np.newaxis, np.newaxis]  # Shape: (3, N, 1, 1)

		# Get dVis and dAis variables
		x6 = self._dAis_dxs[:, :, :, np.newaxis]  # Shape: (3, N, N, 1)
		x10 = self._dVis_dxs[:, :, :, np.newaxis]  # Shape: (3, N, N, 1)
		dVi_dxj = x10  # Shape: (3, N, N, 1)

		x7 = self._dVis_dxs[:, :, np.newaxis, :]  # Shape: (3, N, 1, N)
		dVi_dxk = x7  # Shape: (3, N, 1, N)
		x9 = self._dAis_dxs[:, :, np.newaxis, :]  # Shape: (3, N, 1, N)

		# Second derivatives
		x0 = self._d2Vis_dxixjs  # Shape: (3, N, N, N)
		x2 = x0  # Same as x0
		x3 = self._d2Ais_dxixjs  # Shape: (3, N, N, N)

		# Compute intermediate variables
		x8 = x6 * x7  # Shape: (3, N, N, N)
		x11 = x10 * x9  # Shape: (3, N, N, N)
		x12 = 2.0 * x6 * x9  # Shape: (3, N, N, N)

		x13 = Vi_expanded * x1_expanded  # Shape: (3, N, 1, 1)
		x13_x6 = x13 * x6  # Shape: (3, N, N, 1)
		x14 = x10 - x13_x6  # Shape: (3, N, N, 1)

		# Compute the value
		self._d2lngammas_c_dxixjs = (
				5.0 * qi_expanded * (
						-x1_expanded * x14 * x15_expanded * x9 + x1_expanded * x2 - x11 * x4_expanded
						+ x15_expanded * (
								x1_expanded * x11 + x1_expanded * x8 - x12 * x5_expanded + x13 * x3 - x2
						)
						- x3 * x5_expanded - x4_expanded * x8 + x14 * x7 * Vi_inv2_expanded + Vi_expanded * x12 * Ai_inv3_expanded
				)
				- x0 + x0 / self._Vis[:, :, np.newaxis, np.newaxis] - dVi_dxj * dVi_dxk * Vi_inv2_expanded
		)

		# Assign the computed values to the output array
		return self._d2lngammas_c_dxixjs  # Shape: (3, N, N, N)


	def d2lngammas_r_dxixjs(self):

		try:
			self._d2lnGammas_subgroups_dxixjs
		except AttributeError:
			self.d2lnGammas_subgroups_dxixjs()
		try:
			self._occurrance_matrix
		except AttributeError:
			self.occurrance_matrix()
		self._d2lngammas_r_dxixjs = np.einsum('mi,fjkm->fijk', self._occurrance_matrix, self._d2lnGammas_subgroups_dxixjs)
		return self._d2lngammas_r_dxixjs


	def d2Vis_dxixjs(self):
		try:
			self._rbar
		except AttributeError: 
			self.rbar()
		
		rbar_sum_inv3_2 = 2.0*1/(self._rbar)**3
		rs_i = self._r[:, np.newaxis, np.newaxis]  # (N, 1, 1)
		rs_j = self._r[np.newaxis, :, np.newaxis]  # (1, N, 1)
		rs_k = self._r[np.newaxis, np.newaxis, :]  # (1, 1, N)
		rs_3 = rs_i * rs_j * rs_k 
		self._d2Vis_dxixjs = rbar_sum_inv3_2[:,np.newaxis, np.newaxis,np.newaxis] * rs_3[np.newaxis,:]
		return self._d2Vis_dxixjs


	def d2Ais_dxixjs(self):
		try:
			self._qbar
		except AttributeError: 
			self.qbar()
		
		qbar_sum_inv3_2 = 2.0*1/(self._qbar)**3
		qs_i = self._q[:, np.newaxis, np.newaxis]  # (N, 1, 1)
		qs_j = self._q[np.newaxis, :, np.newaxis]  # (1, N, 1)
		qs_k = self._q[np.newaxis, np.newaxis, :]  # (1, 1, N)
		qs_3 = qs_i * qs_j * qs_k 
		self._d2Ais_dxixjs = qbar_sum_inv3_2[:,np.newaxis, np.newaxis,np.newaxis] * qs_3[np.newaxis,:]
		return self._d2Ais_dxixjs
	

	def Zs(self):
		try:
			self._Thetas
		except AttributeError:
			self.Thetas()
		try:
			self._psis
		except AttributeError:
			self.psis()
		self._Zs = 1/(self._Thetas @ self._psis)
		return self._Zs


	def d2lnGammas_subgroups_dxixjs(self):
		try:
			self._Zs
		except AttributeError: 
			self.Zs()
		try:
			self._Ws
		except AttributeError:
			self.Ws()
		try:
			self._d2Thetas_dxixjs
		except AttributeError:
			self.d2Thetas_dxixjs()

		d2lnGammas_subgroups_dxixjs = np.empty((np.shape(self.z)[0], self.N, self.N, self.N_groups))

		for f in range(np.shape(self.z)[0]):
			for i in range(self.N):
				# Extract the relevant slice for d2Thetas_dxixjs
				d2Thetas_dxixjs_ij = self._d2Thetas_dxixjs[f, i]  # Shape: (N, N_groups)
				
				# Compute vec0 for all j and k simultaneously
				# vec0[j, k] = sum over m of psis[m, k] * d2Thetas_dxixjs_ij[j, m]
				vec0 = np.dot(d2Thetas_dxixjs_ij, self._psis)  # Shape: (N, N_groups)
				
				for j in range(self.N):
					# Extract vectors for the current indices
					vec0_j = vec0[j]  # Shape: (N_groups,)
					d2Thetas_dxixjs_ij_j = d2Thetas_dxixjs_ij[j]  # Shape: (N_groups,)
					Ws_f_i = self._Ws[f, :, i]  # Shape: (N_groups,)
					Ws_f_j = self._Ws[f, :, j]  # Shape: (N_groups,)
					Zs_f = self._Zs[f]  # Shape: (N_groups,)
					Thetas_f = self._Thetas[f]  # Shape: (N_groups,)
					dThetas_dxs_f_i = self._dThetas_dxs[f, :, i]  # Shape: (N_groups,)
					dThetas_dxs_f_j = self._dThetas_dxs[f, :, j]  # Shape: (N_groups,)

					# Compute intermediate variables A and B
					A = 2.0 * Ws_f_i * Ws_f_j * Zs_f - vec0_j  # Shape: (N_groups,)
					B = Ws_f_j * dThetas_dxs_f_i + Ws_f_i * dThetas_dxs_f_j  # Shape: (N_groups,)

					# Compute d for all m simultaneously
					d = d2Thetas_dxixjs_ij_j + Zs_f * (A * Thetas_f - B)  # Shape: (N_groups,)

					# Compute v for all k simultaneously
					v = np.dot(self._psis, d * Zs_f)  # Shape: (N_groups,)

					# Add the remaining term to v
					v += Zs_f * (vec0_j - Ws_f_i * Ws_f_j * Zs_f)

					# Compute the final result
					d2lnGammas_subgroups_dxixjs[f, i, j, :] = -v * self._group_Q

		self._d2lnGammas_subgroups_dxixjs = d2lnGammas_subgroups_dxixjs
		return self._d2lnGammas_subgroups_dxixjs


	def d2Thetas_dxixjs(self):

		try:
			self._F
		except AttributeError:
			self.F()
		try:
			self._G
		except AttributeError:
			self.G()
		try:
			self._sum_occurance_matrix
		except AttributeError: 
			self.sum_occurance_matrix()
		try:
			self._sum_weighted_number
		except AttributeError: 
			self.sum_weighted_number()
		
		QsVSXS = (self._group_Q * self._sum_weighted_number).sum(axis=1)
		QsVSXS_sum_inv = 1.0/QsVSXS
		n2F = -2.0*self._F
		F2_2 = 2.0*self._F*self._F
		QsVSXS_sum_inv2 = 2.0*QsVSXS_sum_inv
		nffVSj = -self._F[:,np.newaxis]*self._sum_occurance_matrix[np.newaxis,:]
		n2FVsK = n2F[:,np.newaxis] * self._sum_occurance_matrix[np.newaxis,:]

		# for vec0 calculation
		# Reshape and expand arrays for broadcasting
		nffVSj_expanded = nffVSj[np.newaxis,:,:]

		# Transpose VSXS to align dimensions
		VSXS_transposed = np.transpose(self._sum_weighted_number[:, :, np.newaxis] , (1, 2, 0)).reshape(len(self._group_Q), np.shape(self.z)[0], 1)

		# Add vs and multiply by Qs
		vec0 = np.sum(self._group_Q[:,np.newaxis,np.newaxis] * (VSXS_transposed * nffVSj_expanded + self._occurrance_matrix.reshape(len(self._group_Q),1,np.shape(self.z)[1])), axis=0)

		# for tot0 calculation
		# Reshape variables to align dimensions for broadcasting
		# Reshaping variables to add singleton dimensions where necessary
		# n2FVsK: (3, 1, 2, 1)
		n2FVsK_expanded = n2FVsK[:, np.newaxis, :, np.newaxis]  # (3, 1, 2, 1)
		# VSXS: (3, 1, 1, 4)
		VSXS_expanded = self._sum_weighted_number[:, np.newaxis, np.newaxis, :]      # (3, 1, 1, 4)
		# VS: (1, 2, 1, 1)
		VS_j_expanded = self._sum_occurance_matrix[np.newaxis, :, np.newaxis, np.newaxis]  # (1, N, 1, 1)
		# VS: (1, 1, 2, 1)
		VS_k_expanded = self._sum_occurance_matrix[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, N, 1)
		# self._occurrance_matrix[n, k]: (1, 1, 2, 4)
		occurrance_matrix_nk_expanded = self._occurrance_matrix.T[np.newaxis, np.newaxis, :, :]  # vs.T shape is (2, 4), transpose vs to get (2, N_groups)
		# vs[n, j]: (1, 2, 1, 4)
		occurrance_matrix_nj_expanded = self._occurrance_matrix.T[np.newaxis, :, np.newaxis, :]  # vs.T shape is (2, 4)
		# Qs: (1, 1, 1, 4)
		Qs_expanded = self._group_Q[np.newaxis, np.newaxis, np.newaxis, :]  # (1, 1, 1, N_groups)

		# Compute the first term: VS[j] * (n2FVsK * VSXS + vs[n, k])
		term1 = VS_j_expanded * (n2FVsK_expanded * VSXS_expanded + occurrance_matrix_nk_expanded)

		# Compute the second term: VS[k] * vs[n, j]
		term2 = VS_k_expanded * occurrance_matrix_nj_expanded

		# Sum both terms
		terms = term1 + term2  # Shape: (3, 2, 2, 4)

		# Multiply by Qs[n]
		terms *= Qs_expanded  # Broadcasting over Qs

		# Sum over n (axis=-1) to get tot0 of shape (3, 2, 2)
		tot0 = np.sum(terms, axis=-1)

		# Multiply by F and QsVSXS_sum_inv
		F_expanded = self._F[:, np.newaxis, np.newaxis]  # (3, 1, 1)
		QsVSXS_sum_inv_expanded = QsVSXS_sum_inv[:, np.newaxis, np.newaxis]  # (3, 1, 1)

		tot0 *= F_expanded * QsVSXS_sum_inv_expanded # shape: (3, 2, 2)

		# Initialize d2Thetas_dxixjs with the appropriate shape
		d2Thetas_dxixjs = np.zeros((np.shape(self.z)[0], self.N, self.N, self.N_groups))

		for f in range(np.shape(self.z)[0]):
			for j in range(self.N):
				VS_j = self._sum_occurance_matrix[j]  # Scalar
				vs_j = self._occurrance_matrix[:, j]  # Shape: (self.N_groups,)
				vec0_fj = vec0[f, j]  # Scalar

				# Shapes for broadcasting
				VS_k = self._sum_occurance_matrix  # Shape: (self.N,)
				vs_k = self._occurrance_matrix  # Shape: (self.N_groups, self.N)
				vec0_fk = vec0[f]  # Shape: (self.N,)

				# Compute terms using broadcasting
				term1 = -self._F[f] * (VS_j * vs_k + VS_k[np.newaxis, :] * vs_j[:, np.newaxis])  # Shape: (N_groups, N)
				term2 = self._sum_weighted_number[f, :, np.newaxis] * tot0[f, j, :]  # Shape: (N_groups, N)
				term3 = F2_2[f] * VS_j * VS_k[np.newaxis, :] * self._sum_weighted_number[f, :, np.newaxis]  # Shape: (N_groups, N)

				# Compute the temp term using broadcasting
				temp = QsVSXS_sum_inv[f] * (
						QsVSXS_sum_inv2[f] * self._sum_weighted_number[f, :, np.newaxis] * vec0_fj * vec0_fk[np.newaxis, :]
						- vs_j[:, np.newaxis] * vec0_fk[np.newaxis, :] - vs_k * vec0_fj
						+ self._F[f] * self._sum_weighted_number[f, :, np.newaxis] * (VS_j * vec0_fk[np.newaxis, :] + VS_k[np.newaxis, :] * vec0_fj)
				)  # Shape: (N_groups, N)

				# Sum all terms
				v = term1 + term2 + term3 + temp  # Shape: (N_groups, N)

				# Compute the final result with broadcasting
				result = (v * self._group_Q[:, np.newaxis] * QsVSXS_sum_inv[f]).T  # Shape: (N, N_groups)

				# Assign the computed values to the result tensor
				d2Thetas_dxixjs[f, j, :, :] = result

		self._d2Thetas_dxixjs = d2Thetas_dxixjs
		return self._d2Thetas_dxixjs


	def d2GE_dxixjs(self):

		try:
			self._dlngammas_c_dxs
		except AttributeError:
			self.dlngammas_c_dxs()
		try:
			self._dlngammas_r_dxs
		except AttributeError:
			self.dlngammas_r_dxs()
		try:
			self._d2lngammas_c_dxixjs
		except AttributeError:
			self.d2lngammas_c_dxixjs()
		try:
			self._d2lngammas_r_dxixjs
		except AttributeError:
			self.d2lngammas_r_dxixjs()

		# Sum the terms and their transposes over the axes
		dGE_initial = self._dlngammas_c_dxs + self._dlngammas_r_dxs + self._dlngammas_c_dxs.transpose(0, 2, 1) + self._dlngammas_r_dxs.transpose(0, 2, 1)

		# Expand the dimensions of z to align for broadcasting
		z_expanded = self.z[:, :, np.newaxis, np.newaxis]  

		# Sum over k (axis=1) after multiplying z with the sum of second derivatives
		sum_over_k = np.sum(z_expanded * (self._d2lngammas_c_dxixjs + self._d2lngammas_r_dxixjs), axis=1)

		# Combine all terms
		self._d2GE_dxixjs = self.Rc * self.T * (dGE_initial + sum_over_k)  # Shape: (3, 2, 2)
		return self._d2GE_dxixjs


	def dmu_dz(self):

		try:
			self._dlngammas_c_dxs
		except AttributeError:
			self.dlngammas_c_dxs()
		try:
			self._dlngammas_r_dxs
		except AttributeError:
			self.dlngammas_r_dxs()
		try:
			self._d2lngammas_c_dxixjs
		except AttributeError:
			self.d2lngammas_c_dxixjs()
		try:
			self._d2lngammas_r_dxixjs
		except AttributeError:
			self.d2lngammas_r_dxixjs()

		# Sum the terms and their transposes over the axes
		dmu_initial = self._dlngammas_c_dxs + self._dlngammas_r_dxs + self._dlngammas_c_dxs.transpose(0, 2, 1) + self._dlngammas_r_dxs.transpose(0, 2, 1) + 1/self.z[:,np.newaxis]

		# Expand the dimensions of z to align for broadcasting
		z_expanded = self.z[:, :, np.newaxis, np.newaxis]  

		# Sum over k (axis=1) after multiplying z with the sum of second derivatives
		sum_over_k = np.sum(z_expanded * (self._d2lngammas_c_dxixjs + self._d2lngammas_r_dxixjs), axis=1)

		# Combine all terms
		self._dmu_dz = self.Rc * self.T * (dmu_initial + sum_over_k)  # Shape: (3, 2, 2)
		return self._dmu_dz
	

	
	def det_Hij(self):
		try:
			self._dmu_dz
		except AttributeError:
			self.dmu_dz()

		if self.z.shape[1] == 2:
			Hij = np.empty((np.shape(self.z)[0], np.shape(self.z)[1]-1, np.shape(self.z)[1]-1))
			n = np.shape(self.z)[1]-1
			for ii in range(n):
				for jj in range(n):
					Hij[:,ii,jj] = self._dmu_dz[:,ii,jj] + self._dmu_dz[:,n,n]
			return np.linalg.det(Hij)

		elif self.z.shape[1] == 3:
			Hij = np.empty((np.shape(self.z)[0], np.shape(self.z)[1]-1, np.shape(self.z)[1]-1))
			n = np.shape(self.z)[1]-1
			for ii in range(n):
				for jj in range(n):
						Hij[:,ii,jj] = self._dmu_dz[:,ii,jj] - self._dmu_dz[:,n,jj] #+ self._dmu_dz[:,n,n] #- self._dmu_dz[:,ii,n] 
			det_Hij = np.linalg.det(Hij)
			return det_Hij

