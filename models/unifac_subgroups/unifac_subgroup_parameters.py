import re, ast, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# read files for creating main group & subgroup objects
_here = Path(__file__).parent

_unif_info = pd.read_csv(_here / "unifac_info.csv")
_unif_il_info = _unif_info.drop(columns=["R", "Q"]).rename(columns={"IL_R": "R", "IL_Q": "Q"})

_unif_subgroups = pd.read_csv(_here / 'unifac_subgroups.csv')
_unif_il_subgroups = pd.read_csv(_here / 'unifac_il_subgroups.csv')

unif = pd.merge(_unif_info, _unif_subgroups, on=['subgroup', 'R', 'Q'], how='inner').drop(columns=["IL_R", "IL_Q"]).set_index("subgroup_id")
unif_il = pd.merge(_unif_il_info, _unif_il_subgroups, on=['subgroup', "R", "Q"], how='inner').set_index("subgroup_id")

class UNIFAC_subgroup:
	__slots__ = ['group_id', 'group', 'main_group_id', 'main_group', 'R', 'Q', 'smarts', 'priority', 'atoms', 'bonds']

	def __repr__(self):   # pragma: no cover
		return f'<{self.group}>'

	def __init__(self, group_id, group, main_group_id, main_group, R, Q, smarts=None,
					priority=None, atoms=None, bonds=None, ):
		
		self.group_id = group_id
		self.group = group
		self.main_group_id = main_group_id
		self.main_group = main_group
		self.R = R
		self.Q = Q
		self.smarts = smarts
		self.priority = priority
		self.atoms = atoms
		self.bonds = bonds

# assigns bonds types
SINGLE_BOND = 'single'
DOUBLE_BOND = 'double'
TRIPLE_BOND = 'triple '
AROMATIC_BOND = 'aromatic'

""" Rules for bonds: All groups that have any any atoms as part of any aromatic ring should have at least one aromatic bond. """

# UFSG[subgroup ID] = (subgroup formula, main group ID, subgroup R, subgroup Q)
# retrieves unifac subgrouops from "thermo_unifac_subgroups.csv"
UFSG = {}
for i in unif.index:
	UFSG[i] = UNIFAC_subgroup(
					group_id=i, 
					group=unif["subgroup"][i], 
					main_group_id=unif["main_group_id"][i], 
					main_group=unif["main_group"][i], 
					R=unif["R"][i], 
					Q=unif["Q"][i], 
					bonds=unif["bonds"][i], 
					atoms=unif["atoms"][i], 
					smarts=unif["smarts"][i], 
	)

UFILSG = {}
for i in unif_il.index:
	UFILSG[i] = UNIFAC_subgroup(
					group_id=i, 
					group=unif_il["subgroup"][i], 
					main_group_id=unif_il["main_group_id"][i], 
					main_group=unif_il["main_group"][i], 
					R=unif_il["R"][i], 
					Q=unif_il["Q"][i], 
					bonds=unif_il["bonds"][i], 
					atoms=unif_il["atoms"][i], 
					smarts=unif_il["smarts"][i], 
	)
# then add subgroups from original unifac
for i in unif.index:
	if  i not in UFILSG.keys():
		UFILSG[i] = UNIFAC_subgroup(
					group_id=i, 
					group=unif["subgroup"][i], 
					main_group_id=unif["main_group_id"][i], 
					main_group=unif["main_group"][i], 
					R=unif["R"][i], 
					Q=unif["Q"][i], 
					bonds=unif["bonds"][i], 
					atoms=unif["atoms"][i], 
					smarts=unif["smarts"][i], 
	)

def priority_from_atoms(atoms, bonds=None):
	priority = 0
	if 'H' in atoms:
		priority += atoms['H']
	if 'C' in atoms:
		priority += atoms['C']*100
	if 'O' in atoms:
		priority += atoms['O']*150
	if 'N' in atoms:
		priority += atoms['N']*175
	if 'Cl' in atoms:
		priority += atoms['Cl']*300
	if 'F' in atoms:
		priority += atoms['F']*400
	if 'Si' in atoms:
		priority += atoms['Si']*200
	if 'S' in atoms:
		priority += atoms['S']*250

	if bonds is not None:
		priority += bonds.get(SINGLE_BOND, 0)*2
		priority += bonds.get(DOUBLE_BOND, 0)*10
		priority += bonds.get(TRIPLE_BOND, 0)*100
		priority += bonds.get(AROMATIC_BOND, 0)*1000
		
	return priority

for group in UFSG.values():
	if group.priority is None:
		if group.atoms is not None:
			# converts str to dict
			if type(group.atoms) == str:
				group.atoms = ast.literal_eval(group.atoms)
			# returns none type if NaN is assigned
			elif np.isnan(group.atoms):
				group.atoms = None
			# converts str to dict
			if type(group.bonds) == str:
				group.bonds = ast.literal_eval(group.bonds)
			# returns none type if NaN is assigned
			elif np.isnan(group.bonds):
				group.bonds = None
			group.priority = priority_from_atoms(group.atoms, group.bonds)


# for interaction parameters, first priority is LLE, then use original unifac

# get dictionary of interaction parameters
UFMG = unif["main_group_id"].tolist()
UFIP = {i: {} for i in UFMG}
# add LLE first
with open(_here / 'unifac_LLE_ip.csv', 'r') as f:
	for line in f:
		maingroup1, maingroup2, ip_A = line.strip('\n').split(',')
		UFIP[int(maingroup1)][int(maingroup2)] = float(ip_A)
# if not in LLE data, then use original UNIFAC params
with open(_here / 'unifac_ip.csv', 'r') as f:
	for line in f:
		maingroup1, maingroup2, ip_A = line.strip('\n').split(',')
		try:
			UFIP[int(maingroup1)][int(maingroup2)]
		except:
			UFIP[int(maingroup1)][int(maingroup2)] = float(ip_A)

# for ionic liquid interaction parameters, priority == ionic liquids, LLE, original 
UFILMG = unif_il["main_group_id"].tolist()
all_MG = np.unique(np.array(UFMG + UFILMG))
UFILIP = {i: {} for i in all_MG}
with open(_here / 'unifac_il_ip.csv', 'r') as f:
	for line in f:
		maingroup1, maingroup2, ip_A = line.strip('\n').split(',')
		UFILIP[int(maingroup1)][int(maingroup2)] = float(ip_A)
# add LLE / original unifac data if they don't already exist
with open(_here / 'unifac_LLE_ip.csv', 'r') as f:
	for line in f:
		try:
			UFILIP[int(maingroup1)][int(maingroup2)]
		except:
			UFILIP[int(maingroup1)][int(maingroup2)] = float(ip_A)
# if not in LLE data, then use original UNIFAC params
with open(_here / 'unifac_ip.csv', 'r') as f:
	for line in f:
		maingroup1, maingroup2, ip_A = line.strip('\n').split(',')
		try:
			UFILIP[int(maingroup1)][int(maingroup2)]
		except:
			UFILIP[int(maingroup1)][int(maingroup2)] = float(ip_A)

