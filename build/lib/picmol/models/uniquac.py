import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

from .unifac_subgroups.fragmentation import Groups
from .cem import PointDisc

def UNIQUAC_R(smiles):
	g = [Groups(smile, "smiles") for smile in smiles]
	r = np.array([gg.unifac.r for gg in g])
	return r

def UNIQUAC_Q(smiles):
	g = [Groups(smile, "smiles") for smile in smiles]
	q = np.array([gg.unifac.q for gg in g])
	return q

def UNIQUAC_RQ(smiles):
	g = [Groups(smile, "smiles") for smile in smiles]
	r = np.array([gg.unifac.r for gg in g])
	q = np.array([gg.unifac.q for gg in g])
	return r, q

def fit_du_to_Hmix(T, Hmix, z, smiles):
	""" fit UNIQUAC interaction parameters """

	N_mols = z.shape[1]
	r, q = UNIQUAC_RQ(smiles)
	
	globals()['r'] = r
	globals()['q'] = q
	globals()['Rc'] = 8.314E-3
	globals()['T'] = T

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
def UNIQUAC_Hmix_fit_func({arg_list}):
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
			T = globals()['T']

			zc = 10
			l = (zc / 2) * (r - q) - (r - 1)
			tau = np.exp(-du / (Rc * T))
			rbar = z @ r
			qbar = z @ q
			# Calculate phi and theta (segment and area fractions)
			phi = z * (1 / rbar[:, None]) * r
			theta = z * (1 / qbar[:, None]) * q
			# Solve for free energy of mixing GM 
			GM_res = -Rc * T * (z * np.log(np.dot(theta , tau))) @ q
			return GM_res
"""
			# Create a local namespace for the function definition
			func_namespace = {}
			# Include '__name__' in the globals dictionary
			exec(func_code, globals(), func_namespace)
			# Retrieve the dynamically created function
			du_function = func_namespace['UNIQUAC_Hmix_fit_func']
			return du_function
	
	ij_combo = 0
	for i in range(N_mols):
		for j in range(N_mols):
			if i < j:
				ij_combo += 1

	UNIQUAC_Hmix_fit = create_du_function(ij_combo)

	popt, pcov = curve_fit(UNIQUAC_Hmix_fit, z, Hmix)

	du = np.zeros((N_mols, N_mols))
	ij = 0
	for i in range(N_mols):
		for j in range(N_mols):
			if i < j:
				du[i,j] = popt[ij]
				du[j,i] = popt[ij]
				ij += 1
		
	return du


def fit_du_to_GM(T, GM, z, smiles):
	""" fit UNIQUAC interaction parameters """

	N_mols = z.shape[1]
	r, q = UNIQUAC_RQ(smiles)
	
	globals()['r'] = r
	globals()['q'] = q
	globals()['Rc'] = 8.314E-3
	globals()['T'] = T

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
			T = globals()['T']

			zc = 10
			l = (zc / 2) * (r - q) - (r - 1)
			tau = np.exp(-du / (Rc * T))
			rbar = z @ r
			qbar = z @ q
			# Calculate phi and theta (segment and area fractions)
			phi = z * (1 / rbar[:, None]) * r
			theta = z * (1 / qbar[:, None]) * q
			# Solve for free energy of mixing GM 
			GM_comb = Rc * T * (np.sum(z * np.log(phi), axis=1) + (zc / 2) * (z * (np.log(theta) - np.log(phi))) @ q)
			GM_res = -Rc * T * (z * np.log(np.dot(theta , tau))) @ q
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

	popt, pcov = curve_fit(UNIQUAC_GM_fit, z, GM)

	du = np.zeros((N_mols, N_mols))
	ij = 0
	for i in range(N_mols):
		for j in range(N_mols):
			if i < j:
				du[i,j] = popt[ij]
				du[j,i] = popt[ij]
				ij += 1
		
	return du



class UNIQUAC:
	
	def __init__(self, IP, smiles, z=None, r=None, q=None):
		self.du = IP
		self.zc = 10
		self.R = 8.314E-3
		self.N = len(smiles)

		num_pts = {
			2:10, 3:7, 4:6, 5:5, 6:4
		}
		self.num_comp = len(smiles)
		self.rec_steps = num_pts[self.num_comp]

		if z is not None:
			self.z = z
		else:
			point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.rec_steps, load=True, store=False)
			self.z = point_disc.points_mfr    
		
		self.smiles = smiles

		if r is not None and q is not None:
			self._r = r
			self._q = q


	def update_z(self, new_value):
		self.z = new_value

	@property
	def r(self):
		try:
			self._r 
		except AttributeError:
			self._r = UNIQUAC_R(self.smiles)
		return self._r

	@property
	def q(self):
		try:
			self._q
		except AttributeError:
			self._q = UNIQUAC_Q(self.smiles)
		return self._q

	@property
	def l(self):
		return (self.zc / 2) * (self.r - self.q) - (self.r - 1)

	@property
	def rbar(self):
		return self.z @ self.r
	
	@property
	def qbar(self):
		return self.z @ self.q
	
	@property
	def lbar(self):
		return self.z @ self.l

	@property
	def rr(self):
		return np.ones((self.z.shape[0], 1)) * self.r

	@property
	def qq(self):
		return np.ones((self.z.shape[0], 1)) * self.q
	
	@property
	def ll(self):
		return np.ones((self.z.shape[0], 1)) * self.l

	@property
	def phi(self):
		return self.z * (1 / self.rbar[:, None]) * self.r

	@property
	def theta(self):
		return self.z * (1 / self.qbar[:, None]) * self.q
	
	def tau(self, T):
		return np.exp(-self.du / (self.R * T))

	def rho(self, T):
		return self.theta @ self.tau(T)

	def GE(self, T):
		'''excess energy'''
		return np.nan_to_num(self.GE_res(T) + self.GE_comb(T))
	
	def GE_res(self, T):
		return -self.R * T * (self.z * np.log(np.dot(self.theta , self.tau(T)))) @ self.q
	
	def GE_comb(self, T):
		ge_comb = self.R * T * (np.sum(self.z * np.log(self.phi/self.z), axis=1) + (self.zc / 2) * (self.z * (np.log(self.theta/self.phi))) @ self.q)
		return ge_comb
	
	def Gid(self, T):
		return self.R * T * np.sum(self.z * np.log(self.z), axis=1)

	def GM(self, T):
		'''mixing free energy'''
		return np.nan_to_num(self.GE(T) + self.Gid(T))
	
	def gammas(self, T):
		'''activity coefficients'''

		theta_tau_j = np.zeros(self.z.shape)
		for i in range(self.z.shape[1]):
			theta_tau = np.zeros(self.z.shape[0])
			for j in range(self.z.shape[1]):
				theta_tau += self.theta[:,j] * self.tau(T)[i,j] / self.rho(T)[:,j]
			theta_tau_j[:,i] = theta_tau

		gammas = np.log(self.phi/self.z) + (self.zc / 2) * self.qq * (np.log(self.theta/self.phi)) + self.ll - (self.phi/self.z) * (self.lbar[:, None]) -  self.qq * np.log(np.dot(self.theta , self.tau(T))) + self.qq - self.qq *  theta_tau_j

		return gammas
	

	def mu(self, T):
		'''calculates the chemical potential'''

		mu_comb = ((np.log(self.phi/self.z)) + (self.zc / 2) * (self.qq * (np.log(self.theta/self.phi))) + self.ll - (self.lbar[:, None] @ np.array([np.ones(np.shape(self.z)[1])])) * self.phi / self.z)

		mu_res = - (self.qq * (np.log(np.dot(self.theta , self.tau(T)))) - self.qq * (1 - (self.theta / (np.dot(self.theta , self.tau(T)))) @ self.tau(T)))

		return self.R * T * (mu_comb + mu_res)
	

	def dGM_dxs(self, T):
		'''calculates the first derivative of mixing free energy'''
		mu = self.mu(T)
		dGM_dxs = np.zeros((self.z.shape[0], self.z.shape[1]-1))
		for i in range(self.z.shape[1]-1):
			dGM_dxs[:,i] = mu[:,i] - mu[:,self.z.shape[1]-1]
		return dGM_dxs


	def dmu_dz(self, T):
		'''calculates the second derivative via analytical functions'''

		ones_2darr = np.array([np.ones(np.shape(self.z)[1])])
		dmu_dz_comb = np.zeros((np.shape(self.z)[0], np.shape(self.z)[1], np.shape(self.z)[1]))

		for ii in range(np.shape(self.z)[1]):
			i_arr = np.array([np.zeros(np.shape(self.z)[1])])
			i_arr[0][ii] = 1

			dphi_dz1 = (self.rbar[:, None] * (self.r[None, :] * i_arr) - self.rr * self.z * self.r[ii]) / (self.rbar[:, None] ** 2 @ ones_2darr)
			dtheta_dz1 = (self.qbar[:, None] * (self.q[None, :] * i_arr) - self.qq * self.z * self.q[ii]) / (self.qbar[:, None] ** 2 @ ones_2darr)

			dmu_comb1_dz1 = ((1 -(self.zc / 2) * self.qq) * dphi_dz1 / self.phi + (self.zc / 2) * self.qq * dtheta_dz1 / self.theta)

			dmu_comb2_dz1 = -((self.lbar[:, None] @ ones_2darr) * (dphi_dz1 / self.z - self.phi * (np.ones((self.z.shape[0], 1)) @ i_arr) / self.z ** 2) + (self.phi / self.z) * self.l[ii])

			dmu_dzi = dmu_comb1_dz1 + dmu_comb2_dz1
			dmu_dz_comb[:,ii,:] = dmu_dzi
		
		dmu_dz_res = np.zeros((np.shape(self.z)[0], np.shape(self.z)[1], np.shape(self.z)[1]))
		for ii in range(np.shape(self.z)[1]):
			for jj in range(np.shape(self.z)[1]):
				dmu_dz_res[:,ii,jj] = ((self.q[ii]*self.q[jj])/(self.qbar)) * (1 - (self.tau(T)[jj][ii]/self.rho(T)[:,ii]) - (self.tau(T)[ii][jj]/self.rho(T)[:,jj]) + (((self.theta * self.tau(T)[ii] )/ self.rho(T)**2) @ self.tau(T)[jj]))
		
		dmu_dz = self.R * T * (dmu_dz_comb + dmu_dz_res)
		return dmu_dz


	def Hij(self, T):
		dmu_dz = self.dmu_dz(T)
		Hij = np.empty((np.shape(self.z)[0], np.shape(self.z)[1]-1, np.shape(self.z)[1]-1))
		n = np.shape(self.z)[1]-1
		for ii in range(np.shape(self.z)[1]-1):
			for jj in range(np.shape(self.z)[1]-1):
				Hij[:,ii,jj] = dmu_dz[:,ii,jj]-dmu_dz[:,ii,n]-dmu_dz[:,n,jj]+dmu_dz[:,n,n]
		return Hij
	

	def det_Hij(self, T):
		return np.linalg.det(self.Hij(T))