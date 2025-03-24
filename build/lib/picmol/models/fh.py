import numpy as np
import scipy.optimize as sco 
from scipy.optimize import curve_fit
from pathlib import Path
import pandas as pd
from functools import partial

def get_molelecular_properties(smiles):
	file = Path(__file__).parent.parent / "data" / "molecular_properties.csv"
	df = pd.read_csv(file).set_index("smiles")
	return df.loc[smiles, :]




class FH:

	def __init__(self, smiles: list, IP=None):
		""" for binary mixtures only!! """
		self.R = 8.314E-3

		self.V0 = get_molelecular_properties(smiles)["molar_vol"].to_numpy()
		self.V1, self.V2 = self.V0
		self.N0 = [self.V0[i]/min(self.V0) for i in range(len(self.V0))]
		self.N1, self.N2 = self.N0

		self.z = np.load(Path(__file__).parent / "molfr_matrix" / f"{len(smiles)}_molfr_pts.npy")


	def load_thermo_data(self, phi, Smix, Hmix):
		self.Smix = Smix
		self.Hmix = Hmix
		self.phi = phi

	def fit_chi(self, T):
		G = self.Hmix - T * self.Smix

		def GM_fit(x, chi, Tx):
			""" for fitting chi """
			return 8.314E-3 * Tx * (x * np.log(x)/self.N1 + (1-x) * np.log(1-x)/self.N2) + chi*x*(1-x)

		GM_fixed_T = partial(GM_fit, Tx=T)

		fit, pcov = curve_fit(GM_fixed_T, xdata=self.phi, ydata=G)
		chi = fit[0]
		return chi

	def convert_vol2mol(self, phi):
		return phi * (self.V2/self.V1)/(1 - phi + phi*(self.V2/self.V1))

	@property
	def phic(self):
		'''vol fraction of solute at critical pt'''
		return 1/(np.sqrt(self.N1) + np.sqrt(self.N2))

	@property
	def xc(self):
		'''mol fraction of solute at critical pt'''
		return self.phic * (self.V2/self.V1)/(1 - self.phic + self.phic*(self.V2/self.V1))

	@property
	def chic(self):
		'''chi at critical point'''
		return 0.5 * (1/np.sqrt(self.N1) + 1/np.sqrt(self.N1))**2

	def GM(self, x, T):
		'''mixing free energy'''
		chi = self.fit_chi(T)
		return self.R * T * (x * np.log(x)/self.N1 + (1-x) * np.log(1-x)/self.N2) + chi*x*(1-x)
	
	def dGM_dxs(self, x, T):
		'''1st deriv. of mixing free energy'''
		chi = self.fit_chi(T)
		return self.R * T * ((np.log(x) + 1)/self.N1 - (np.log(1-x) + 1)/self.N2) + chi*(1-2*x)
	
	def det_Hij(self, x, T):
		'''2nd deriv. of mixing free energy'''
		chi = self.fit_chi(T)
		return self.R * T * (1/self.N1/x + 1/self.N2/(1-x)) - 2*chi
	
	def Fall(self, x, x0, T):
		'''system free energy of binary mixture'''
		x1, x2 = x[0], x[1]
		dc = (x0-x2)/(x1-x2)
		return dc*self.GM(x1, T) + (1-dc)*self.GM(x2, T)

	def Jall(self, x, x0, T):
		'''3D Jacobian vector for minimization'''
		x1, x2 = x[0], x[1]
		dc = (x0-x2)/(x1-x2)
		f1 = self.GM(x1, T)
		f2 = self.GM(x2, T)
		df1 = self.dGM_dxs(x1, T)
		df2 = self.dGM_dxs(x2, T)
		
		return np.array([-dc/(x1-x2)*(f1-f2) + dc*df1, 
						(1-dc)/(x1-x2)*(f2-f1)+(1-dc)*df2])

	def ps_calc(self, x0, T):
		'''calculate spinodals and binodals'''
		sp1 = sco.brenth(self.det_Hij, 1e-5, x0, args=(T))  
		sp2 = sco.brenth(self.det_Hij, x0, 0.99, args=(T))

		# Minimize system free energy fall defined above to obtain the binodal volume fractions x
		# via L-BFGS-B algorithm
		bis = sco.minimize(self.Fall, x0=(0.01,0.9), args=(x0, T), 
												jac=self.Jall, bounds=((1e-5, sp1),(sp2,0.99)), method='L-BFGS-B')

		# all other thermo packages return mol frac--conver vol2mol frac
		sp1 = self.convert_vol2mol(sp1)
		sp2 = self.convert_vol2mol(sp2)
		bi1 = self.convert_vol2mol(bis.x[0])
		bi2 = self.convert_vol2mol(bis.x[1])

		return sp1, sp2, bi1, bi2 
