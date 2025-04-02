import os



def main():
    # creater parser object
  parser = argparse.ArgumentParser(description='command line interface for calculating KBI from number fluctuations, assumes file name: [system]_[ensemble][suffix].[filetype], for .gro, .xtc, .mdp files, top file: [system].top')

  parser.add_argument('--ensemble', type=str, default='npt', help='ensemble suffix to follow sys_name (default: npt)')
  parser.add_argument('--suffix', type=str, default='', help='suffix after ensemble (default: '')')
  parser.add_argument('--project_path', type=str, default=os.getcwd(), help='path to project for KBI analysis (default: current working directory)')
  parser.add_argument('--sys', type=str, help='individual system in project')
  parser.add_argument('--allocation', type=str, help='project for allocation hours')
  parser.add_argument('--jobsub', type=str, default='true', choices=['true', 'false'], help='submit runscripts? (default: true)')
  parser.add_argument('--start_time', default='100', help='start time (ns) for calculation')
  parser.add_argument('--end_time', default='600', help='end time (ns) for calculation')
  parser.add_argument('--dt', default='10', help='time interval (ns) between snapshots')

  args = parser.parse_args()


  with open(f'{args.project_path}/{args.sys}/runscript_kbi_{args.sys}', 'w') as f:
    f.write(f"""
#PBS -l select=1:ncpus=128:mpiprocs=64:ompthreads=8
#PBS -A {args.allocation}
#PBS -l walltime=48:00:00
#PBS -N kbi_{args.sys}

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=8
NNODES=`wc -l < $PBS_NODEFILE`
echo "NNODES=" $NNODES
export OMP_NUM_THREADS=$NCPUS
module purge
module load anaconda3
source ~/.bash_profile
conda activate picmol
module add gromacs/2023.2
module load gcc/13.2.0
module load openmpi/4.1.6-gcc-13.2.0

cd {args.sys}

picmol-kbi_inf --project_path {args.project_path} --systems {args.sys} --ensemble {args.ensemble} --suffix {args.suffix} --start_time {args.start_time} --end_time {args.end_time} --dt {args.dt}
    """)
  if args.jobsub.lower() == 'true':
    os.chdir(f'{args.project_path}/{args.sys}')
    os.system(f'qsub runscript_kbi_{args.sys}')
    os.chdir(args.project_path)



if __name__ == "__main__":
  main()