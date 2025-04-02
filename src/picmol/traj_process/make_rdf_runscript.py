import numpy as np
import os
import subprocess
from pathlib import Path
import argparse
import sys as sys_arg
import pandas as pd

def main():

  parser = argparse.ArgumentParser(description='command line interface for generating pbs rdf runscripts, assumes file name: [system]_[ensemble][suffix].[filetype], for .gro, .xtc, .mdp files, top file: [system].top')

  parser.add_argument('--ensemble', type=str, default='npt', help='ensemble suffix to follow sys_name (default: npt)')
  parser.add_argument('--suffix', type=str, default='', help='suffix after ensemble (default: '')')
  parser.add_argument('--allocation', type=str, help='project for allocation hours')
  parser.add_argument('--systems', default=[sys for sys in os.listdir(os.getcwd()) if os.path.isdir(f"{os.getcwd()}/{sys}")], help='list of systems, delimiter="," (default: all in current directory)')
  parser.add_argument('--rdf_dir', type=str, default='rdf_files', help='name for rdf directory (default: rdf_files)')
  parser.add_argument('--jobsub', type=str, default='true', choices=['true', 'false'], help='submit runscripts? (default: true)')
  parser.add_argument('--start_time', default='100', help='start time (ns) for rdf calculation (default: 100)')
  parser.add_argument('--end_time', default='600', help='end time (ns) for rdf calculation (default: 600)')
  parser.add_argument('--dt', default='0.01', help='time interval (ns) for rdf calculation (default: 0.01)')
  parser.add_argument('--view_ndx', default='false', choices=['true', 'false'], help='print out ndx mapping')
  parser.add_argument('--add_ndx', default='false', help='add element to ndx mapping, delimeter="," (ex: TODGA,O1)')
  parser.add_argument('--clear_ndx', default='false', choices=['true', 'false'], help='clear entire ndx mapping')

  args = parser.parse_args()

  if args.view_ndx.lower() == 'true':
    ndx_df = pd.read_csv(Path(__file__).parent / "ndx_mapped.txt")
    print(ndx_df)
    sys_arg.exit() 

  if args.add_ndx.lower() != 'false':
    # extract new molecule info
    new_mol_name, new_mol_atom = args.add_ndx.split(',')
    # append to existing file
    with open(Path(__file__).parent / "ndx_mapped.txt", 'a') as f:
      f.write(f"{new_mol_name},{new_mol_atom}\n")
    sys_arg.exit() 

  if args.clear_ndx.lower() == 'true':
    with open(Path(__file__).parent / "ndx_mapped.txt", "w") as f:
      f.write('molecule,atom_index\n')
    sys_arg.exit() 


  if type(args.systems) == list:
    systems = args.systems
  else:
    systems = args.systems.split(',')

  if len(systems) < 0:
    systems = [os.path.basename(os.getcwd())]

  for sys_name in systems:
    if sys_name not in os.path.basename(os.getcwd()):
      os.chdir(sys_name)

    # get smallest possible sampling interval from production mdp
    with open(f"{sys_name}_{args.ensemble}{args.suffix}.mdp", "r") as f:
      lines = f.readlines()
      f.close()
    for line in lines:
      if line.strip().split()[0] == "dt":
        min_dt = line.strip().split()[-1]
      elif line.strip().split()[0] == "nsteps":
        total_steps = line.strip().split()[-1]
      elif line.strip().split()[0].split('-')[0] == 'nstxout':
        step = line.strip().split()[-1]
    dt = float(min_dt)*float(step)
    total_time = float(min_dt)*float(total_steps)

    t_sta = int(float(args.start_time) * 1000)
    t_end = int(float(args.end_time) * 1000)
    rdf_dt = int(float(args.dt) * 1000)

    mols_present = []
    with open(f'{sys_name}.top', 'r') as top:
      lines = top.readlines()
      molecules_ind = [l+1 for l,line in enumerate(lines) if "molecules" in line][0]
      for line in lines[molecules_ind:]:
        line = line.strip()
        if len(line.split()) > 0:
          mols_present.append(str(line.split()[0]))
    # store index_file codes for each molecule
    mol_ind = [] 
    mol_dict = {}
    for m in range(2,len(mols_present)+2):
      mol = mols_present[m-2]
      mol_ind.append(m+len(mols_present))
      mol_dict[mols_present[m-2]] = m+len(mols_present)

    ndx_dict = {}
    for m, mol in enumerate(mols_present):
      ndx_dict[mol] = m+2 # 1st molecule index starts at 2

    # if index file doesn't exist, create one
    if os.path.exists(f"{sys_name}_com.ndx") == False:
      com_dict = {}
      with open(Path(__file__).parent / 'ndx_mapped.txt', 'r') as ndx:
        for line in ndx:
          line = line.strip()
          if line.split(',')[0] in mols_present:
            com_dict[line.split(',')[0]] = line.split(',')[1]
      # if molecule ceneter-of-mass doesn't exist exit the script
      com_not_found_filter = np.ones(len(mols_present))
      for m, mol in enumerate(mols_present):
        if mol in com_dict.keys():
          com_not_found_filter[m] = 0
      try:
        com_not_found = mols_present[bool(com_not_found_filter)]
      except Exception as e:
        print(f"Error {e}, center-of-mass of molecules are not defined in ndx_mapped.txt")
      ndx_cmd = ''
      for mol in mols_present:
        ndx_cmd += f'{ndx_dict[mol]} & a {com_dict[mol]}\n'
      ndx_cmd += 'q\n'
      # try with gromacs (i.e., in conda environment)
      try:
        gmx_ndx_cmd = ("gmx", "make_ndx", "-f", f"{sys_name}_{args.ensemble}{args.suffix}.gro", "-o", f"{sys_name}_com.ndx")
        make_ndx = subprocess.Popen(gmx_ndx_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdin = make_ndx.communicate(ndx_cmd)
      # otherwise try with gmx_mpi on cluster
      except:
        gmx_ndx_cmd = ("gmx_mpi", "make_ndx", "-f", f"{sys_name}_{args.ensemble}{args.suffix}.gro", "-o", f"{sys_name}_com.ndx")
        make_ndx = subprocess.Popen(gmx_ndx_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdin = make_ndx.communicate(ndx_cmd)

    unique_mol_comb = 0
    for i, mol_1 in enumerate(mols_present):
      for j, mol_2 in enumerate(mols_present):
        if i <= j:
          unique_mol_comb += 1

    # create rdf file directory
    if os.path.exists(args.rdf_dir) == False:
      os.mkdir(args.rdf_dir)

    with open(f'runscript_rdf_{sys_name}', 'w') as f:
      f.write(f"""
  #!/bin/bash
  #PBS -l select=1:ncpus=128:mpiprocs=100:ompthreads=8
  #PBS -A {args.allocation}
  #PBS -l walltime=24:00:00
  #PBS -N {sys_name}_rdf

  cd $PBS_O_WORKDIR
  export OMP_NUM_THREADS=8
  module purge
  module add gromacs/2023.2
  module load gcc/13.2.0
  module load openmpi/4.1.6-gcc-13.2.0

  cd {sys_name}

      """)

      for i, mol_1 in enumerate(mols_present):
        for j, mol_2 in enumerate(mols_present):
          if i <= j:
            f.write(f"""
mpirun -np 1 gmx_mpi rdf -f {sys_name}_{args.ensemble}{args.suffix}.xtc -s {sys_name}_{args.ensemble}{args.suffix}.tpr -n {sys_name}_com.ndx -b {t_sta} -e {t_end} -dt {rdf_dt} -ref {mol_ind[i]} -sel {mol_ind[j]} -o {args.rdf_dir}/rdf_{mol_1}_{mol_2}.xvg  
        """)
    f.write(f"""

wait
      """)

    if args.jobsub.lower() == 'true':
      os.system(f'qsub runscript_rdf_{sys_name}')

    os.chdir('../')


if __name__ == "__main__":
  main()