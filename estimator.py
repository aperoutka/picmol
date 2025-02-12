
from .thermo_model import ThermoModel

class PropertyCalculator:

	def __init__(self, model: ThermoModel):
		""" for estimating bulk thermodynamic properties """
		# liquid density, vapor pressure, isothermal compressability, viscosity, heat capacity
		# these are thermodynamic identities to free energy derivatives

