import os, warnings, subprocess, math
import sys as sys_arg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import mdtraj as md
import argparse


def count_molecules_in_grid(sys, ensemble, suffix, grid_spacing, ns_start=100, ns_end=600, ns_dt=100):
  """
  Superimposes a 3D grid onto simulation boxes, counts unique resnames in each cell,
  and stores the counts in a NumPy array.

  Returns:
      np.ndarray: Array with shape (num_gro_files, num_resnames, x, y, z) containing resname counts.
  """
  if os.path.exists('gro_files') == False:
    os.mkdir('gro_files')
  gro_directory = 'gro_files'

  # check that dir is made
  if os.path.exists('kbi_results') == False:
    os.mkdir('kbi_results')
  kbi_save = 'kbi_results'

  # convert times to ps
  times_ns = np.arange(ns_start, ns_end, ns_dt)
  times_ps = 1000 * times_ns
  dt_ps = ns_dt * 1000

  # first check if gro files are made.
  gro_files = []
  for i, time in enumerate(times_ps):
    print(time)
    gro_file = f"{sys}_npt_{int(times_ns[i])}ns.gro"
    if os.path.exists(f"{gro_directory}/{gro_file}") == False:
      os.system(f"echo 0 | gmx trjconv -f {sys}_{ensemble}{suffix}.xtc -s {sys}_{ensemble}{suffix}.gro -b {int(time)} -e {int(time)+1} -dt {dt_ps} -o {gro_directory}/{gro_file}")
    gro_files.append(gro_file)

  num_gro_files = len(gro_files)
  traj = md.load(os.path.join(gro_directory, gro_files[0]))

  box_vectors = traj.unitcell_vectors[0]
  box_lengths = np.array([np.linalg.norm(box_vectors[0]),
                          np.linalg.norm(box_vectors[1]),
                          np.linalg.norm(box_vectors[2])])

  nx = int(np.ceil(box_lengths[0] / grid_spacing))
  ny = int(np.ceil(box_lengths[1] / grid_spacing))
  nz = int(np.ceil(box_lengths[2] / grid_spacing))

  resnames = sorted(list(set([res.name for res in traj.topology.residues]))) # Get unique resnames
  num_resnames = len(resnames)
  all_counts = np.zeros((num_gro_files, num_resnames, nx, ny, nz), dtype=int)

  for i, gro_file in enumerate(gro_files):
    gro_path = os.path.join(gro_directory, gro_file)
    traj = md.load(gro_path)
    positions = traj.xyz[0]
    box_vectors = traj.unitcell_vectors[0]
    box_lengths = np.array([np.linalg.norm(box_vectors[0]),
                            np.linalg.norm(box_vectors[1]),
                            np.linalg.norm(box_vectors[2])])

    # Adjust grid dimensions to match the current box.
    nx = int(np.ceil(box_lengths[0] / grid_spacing))
    ny = int(np.ceil(box_lengths[1] / grid_spacing))
    nz = int(np.ceil(box_lengths[2] / grid_spacing))

    all_counts_frame = np.zeros((num_resnames, nx, ny, nz), dtype=int)

    for residue_index, residue in enumerate(traj.topology.residues):
      residue_atoms = traj.topology.residue(residue_index).atoms
      residue_positions = positions[[atom.index for atom in residue_atoms]]
      residue_center = np.mean(residue_positions, axis=0)

      grid_x = int(np.floor(residue_center[0] / grid_spacing)) % nx
      grid_y = int(np.floor(residue_center[1] / grid_spacing)) % ny
      grid_z = int(np.floor(residue_center[2] / grid_spacing)) % nz

      resname_index = resnames.index(residue.name) #find the index of the resname

      all_counts_frame[resname_index, grid_x, grid_y, grid_z] += 1

    all_counts[i] = all_counts_frame
  
  output_filename = os.path.join('kbi_results', f"grid_counts_{grid_spacing:.1f}_nm.npy")
  np.save(output_filename, all_counts)
  return all_counts

def extract_resnames(sys, ensemble, suffix):
  # get unique residue names from .gro file
  topology = md.load_topology(f"{sys}_{ensemble}{suffix}.gro")
  residue_names = sorted(list(set([res.name for res in topology.residues]))) 
  return residue_names

def calculate_Gij_V(sys, ensemble, suffix, results):
  # get averages from results
  resnames = extract_resnames(sys, ensemble, suffix)
  V = grid_spacing**3
  Gij = np.zeros((len(resnames), len(resnames)))
  for i, mol_i in enumerate(resnames):
    Ni_arr = results[:,i]
    Ni_avg = Ni_arr.mean(axis=0)
    for j, mol_j in enumerate(resnames):
      Nj_arr = results[:,j]
      Nj_avg = Nj_arr.mean(axis=0)
      kd_ij = int(i==j)
      NiNj_avg = (Ni_arr * Nj_arr).mean(axis=0)
      Gij_cell = V*((NiNj_avg-Ni_avg*Nj_avg)/(Ni_avg*Nj_avg) - kd_ij/Ni_avg)
      Gij_cell_nan_filtered = Gij_cell[~np.isnan(Gij_cell)]
      Gij[i,j] = Gij_cell_nan_filtered.mean()
  return Gij

def calculate_Gij_inf(sys, ensemble, suffix, Gij_arr, vr):
  '''convert to Gij as V -> inf'''
  resnames = extract_resnames(sys, ensemble, suffix)
  l_Gij = Gij_arr * vr[:,np.newaxis,np.newaxis]

  Gij_inf_arr = np.zeros((len(resnames), len(resnames)))
  for i, mol_i in enumerate(resnames):
    for j, mol_j in enumerate(resnames):
      Gij_inf, b = np.polyfit(vr[:4], l_Gij[:4, i, j], 1)
      Gij_inf_arr[i,j] = Gij_inf
  return Gij_inf_arr


def main():
  # creater parser object
  parser = argparse.ArgumentParser(description='command line interface for calculating KBI from number fluctuations, assumes file name: [system]_[ensemble][suffix].[filetype], for .gro, .xtc, .mdp files, top file: [system].top')

  parser.add_argument('--ensemble', type=str, default='npt', help='ensemble suffix to follow sys_name (default: npt)')
  parser.add_argument('--suffix', type=str, default='', help='suffix after ensemble (default: '')')
  parser.add_argument('--project_path', type=str, default=os.getcwd(), help='path to project for KBI analysis (default: current working directory)')
  parser.add_argument('--sys', type=str, help='individual system in project')
  parser.add_argument('--start_time', default='100', help='start time (ns) for calculation')
  parser.add_argument('--end_time', default='600', help='end time (ns) for calculation')
  parser.add_argument('--dt', default='10', help='time interval (ns) between snapshots')

  args = parser.parse_args()
  

  prj = os.path.basename(args.project_path)
  os.chdir(f'{args.project_path}/{args.sys}/')

  t_sta_ns = float(args.start_time)
  t_end_ns = float(args.end_time)
  dt_ns = float(dt)
      
  # get smallest box volume
  if os.path.exists('volume.xvg') == False:
    os.system(f"echo volume | gmx energy -f {args.sys}_{args.ensemble}{args.suffix}.edr -o volume.xvg")
  time, V0 = np.loadtxt('volume.xvg', comments=["@","#"], unpack=True)
  start_idx = np.abs(time - t_sta_ps).argmin()
  end_idx = np.abs(time - t_end_ps).argmin()
  V0_filtered = V0[start_idx:end_idx]
  V0_min = V0_filtered.min()

  # get side length corresponding to min box volume
  l = (V0_min)**(1/3)

  # get spacing --> evaluate particle number fluctuations at various fractions of l
  lambda_spacing = np.array([0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 1])
  spacing_vals = lambda_spacing * l

  # get kbi values for each interaction
  Gij_V = []
  for grid_spacing in spacing_vals:
    print("grid spacing: ", round(grid_spacing, 2))
    results = count_molecules_in_grid(args.sys, grid_spacing, ns_start=t_sta_ns, ns_end=t_end_ns, ns_dt=dt_ns)
    Gij_V += [calculate_Gij_V(args.sys, args.ensemble, args.suffix, results)]
  # convert to array
  Gij_V = np.array(Gij_V)
  # save this
  np.save(f'Gij_V.npy', Gij_V)

  # calculate Gij as V->inf and create .txt file
  V_cell = spacing_vals**3 # volumes of grid cells
  vr = (V_cell/V0_min)**(1/3) # volume ratio of grid cells to box volume, but in lengths.
  Gij_inf = calculate_Gij_inf(args.sys, args.ensemble, args.suffix, Gij_V, vr)
  # and save
  np.save(f'Gij_inf.npy', Gij_inf)


if __name__ == "__main__":
  main()




