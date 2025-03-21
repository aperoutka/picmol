import numpy as np
from pathlib import Path

from .cem import PointDisc

class NRTL:

	def __init__(self, IP, smiles=None, z=None):
		""" NRTL for binary mixtures only!! """
		self.R = 8.314E-3
		self.tau12 = IP["tau12"]
		self.tau21 = IP["tau21"]
		self.alpha = 0.2

		self.num_comp = 2
		self.rec_steps = 10

		if z is not None:
			self.z = z
		else:
			point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.rec_steps, load=True, store=False)
			self.z = point_disc.points_mfr


	def update_z(self, new_value):
		self.z = new_value
		

	def gammas(self, T):
		G12 = np.exp(-self.alpha*self.tau12/(self.R*T))
		G21 = np.exp(-self.alpha*self.tau21/(self.R*T))
		x1 = self.z[:,0]
		x2 = self.z[:,1]
		gamma_1 = (x2**2) * (self.tau21 * (G21 / (x1 + x2 * G21))**2 + self.tau12*G12/((x2 + x1*G12)**2))
		gamma_2 = (x1**2) * (self.tau12 * (G12 / (x2 + x1 * G12))**2 + self.tau21*G21/((x1 + x2*G21)**2))
		return np.array([gamma_1, gamma_2]).T


	def GM(self, T):
		"""
		Calculates the mixing free energy per mole for a binary mixture.
		"""
		G12 = np.exp(-self.alpha*self.tau12/(self.R*T))
		G21 = np.exp(-self.alpha*self.tau21/(self.R*T))
		x1 = self.z[:,0]
		x2 = self.z[:,1]
		G_ex = -self.R * T * (x1 * x2 * (self.tau21 * G21/(x1 + x2 * G21) + self.tau12 * G12 / (x2 + x1 * G12))) 
		G_id = self.R * T * (x1 * np.log(x1) + x2 * np.log(x2))
		return G_ex + G_id

	def dGM_dxs(self, T):
		"""
		Calculates the derivative of mixing free energy with respect to composition
		"""
		G12 = np.exp(-self.alpha*self.tau12/(self.R*T))
		G21 = np.exp(-self.alpha*self.tau21/(self.R*T))
		x1 = self.z[:,0]
		x2 = self.z[:,1]
		n = self.R * T * self.tau21 * G21
		m = self.R * T * self.tau12 * G12

		return -1*((n*G21 - 2*n*G21*x1 - n*(1-G21)*(x1**2))/((G21 + (1-G21)*x1)**2) + (m - 2*m*x1 - m*(G12-1)*(x1**2))/((1 + (G12-1)*x1)**2)) + self.R*T*(np.log(x1) - np.log(x2))


	def det_Hij(self, T):
		"""
		Calculates the second derivative of mixing free energy with respect to composition
		"""
		G12 = np.exp(-self.alpha*self.tau12/(self.R*T))
		G21 = np.exp(-self.alpha*self.tau21/(self.R*T))
		x1 = self.z[:,0]
		x2 = self.z[:,1]
		n = self.R * T * self.tau21 * G21
		m = self.R * T * self.tau12 * G12
		return 2 * ((n*G21/((G21 + (1-G21)*x1)**3)) + (m*G12/((1+(G12-1)*x1)**3))) + self.R*T*(1/x1 + 1/x2)


