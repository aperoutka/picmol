import os
from pathlib import Path

from .point_discretization import PointDisc
from .lle import MiscibilityAnalysis

def make_point_discretizations():
	'''make point discretization if does not exist'''

	# create point discretizations if necessary
	discretizations_todo = [[2, 8], [3, 7], [3,8],[3,10], [4, 6], [5, 5], [6, 4]]

	discretization_path = os.path.join(Path(__file__).parent.parent, "discretization")
	if not os.path.isdir(discretization_path):
		os.mkdir(discretization_path)

	for todo_el in discretizations_todo:
		filename = os.path.join(discretization_path, str(todo_el[0]) + "_" + str(todo_el[1]))
		if not os.path.isdir(filename):
			PointDisc(num_comp=todo_el[0], recursion_steps=todo_el[1], load=False, store=True)

				
class CEM:
	"""
	Takes names and some other inputs and sets all the parameters, which are needed to set
	up a lle analysis (for example the interactions parameters are immediately processed to the
	relevant gE_model here). The lle analysis is then executed.
	"""
	def __init__(self, smiles: list, T, gE_model_name: str, thermo_model, IP=None):
		# make point discretizations if they do not exist
		make_point_discretizations()
			
		self.smiles = smiles
		self.num_comp = len(smiles)
		self.T = T

		num_pts = {
			2: 10,
			3: 8,
			4: 6,
			5: 5,
			6: 4,
		}
		self.point_discretization_rec_steps = num_pts[self.num_comp]

		self.point_disc = PointDisc(num_comp=self.num_comp, recursion_steps=self.point_discretization_rec_steps, load=True, store=False)
		
		# create thermodynamic model
		if gE_model_name.lower() == "unifac":
			self.gE_model = thermo_model(z=self.point_disc.points_mfr, T=self.T, smiles=smiles)  

		elif gE_model_name.lower() == "uniquac":
			self.gE_model = thermo_model(z=self.point_disc.points_mfr, T=self.T, smiles=smiles, du=IP)  

		lle_analysis = MiscibilityAnalysis(self.point_disc, self.gE_model, self.T, self.num_comp)
		
		# rows = unique pts, cols = components
		self.binodal_matrix_molfrac = lle_analysis.compute_phase_eq_molfrac()
		self.binodal_matrix_cartcoords = lle_analysis.compute_phase_eq_cartcoord()
		self.binodal_matrix_inds = lle_analysis.compute_phase_eq_indices()

