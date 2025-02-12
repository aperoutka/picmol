import numpy as np
import pandas as pd
from pathlib import Path


from .unifac_subgroups.fragmentation import Groups

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


class UNIQUAC:
	
	def __init__(self, IP, smiles, z=None):
		self.du = IP
		self.zc = 10
		self.R = 8.314E-3
		self.N = len(smiles)
		if z is not None:
			self.z = z
		else:
			self.z = np.load(Path(__file__).parent / "molfr_matrix" / f"{len(smiles)}_molfr_pts.npy")
		self.smiles = smiles

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
		GE_comb = self.R * T * (np.sum(self.z * np.log(self.phi/self.z), axis=1) + (self.zc / 2) * (self.z * (np.log(self.theta) - np.log(self.phi))) @ self.q)
		GE_res = -self.R * T * (self.z * np.log(np.dot(self.theta , self.tau(T)))) @ self.q
		return np.nan_to_num(GE_comb + GE_res)
	
	def GE_res(self, T):
		GE_res = -self.R * T * (self.z * np.log(np.dot(self.theta , self.tau(T)))) @ self.q
		return GE_res
	
	def GE_comb(self, T):
		GE_comb = self.R * T * (np.sum(self.z * np.log(self.phi/self.z), axis=1) + (self.zc / 2) * (self.z * (np.log(self.theta) - np.log(self.phi))) @ self.q)
		return GE_comb
	
	def Gid(self, T):
		return self.R * T * np.sum(self.z * np.log(self.z), axis=1)

	def GM(self, T):
		'''mixing free energy'''
		GM = self.GE(T) + self.R * T * np.sum(self.z * np.log(self.z), axis=1)
		return np.nan_to_num(GM)
	

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

		mu_comb = (np.log(self.phi) + (self.zc / 2) * (self.qq * (np.log(self.theta) - np.log(self.phi))) + self.ll - (self.lbar[:, None] @ np.array([np.ones(np.shape(self.z)[1])])) * self.phi / self.z)

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

			dmu_comb1_dz1 = ((1 - (self.zc / 2) * self.qq) * dphi_dz1 / self.phi + (self.zc / 2) * self.qq * dtheta_dz1 / self.theta)
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