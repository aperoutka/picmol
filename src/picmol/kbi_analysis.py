import os
import sys as sys_arg
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

sys_arg.path.append('/Users/b324115/Library/CloudStorage/Box-Box/critical_phenomena_saxs/allisons_data')
from picmol import ThermoModel, KBI, KBIPlotter, PhaseDiagramPlotter


known_options = [
  '--prj_path',
  '--pure_component_path',
  '--rdf_dir',
  '--kbi_method',
  '--start_time',
  '--end_time',
  '--thermo_model',
  '--help'
]

shorthand_options = [
  '-s',
  '-p',
  '-r',
  '-k',
  '-b',
  '-e',
  '-m',
  '-h'
]

def printhelp():
  print(f'''

WELCOME TO PICMOL: A PHASE INSTABILITY CALCULATOR FOR MOLECULAR DESIGN

This is a workflow for performing KBI analysis and generating LLE phase calculations

KBI workflow requires the following:
  - organization of files: project --> systems --> rdf_directory
  - pure components are stored in a separate directory
  - in each system requires a rdf directory, .edr file for npt production, and .top file
  - the name of rdf directory should be the same for all systems
  - rdf files contain the names of molecule_ids from simulation in rdf calculation
  - top file: system.top
  - edr file: contains system, npt, and .edr

Use of this script: 
  python kbi_analysis.py 
  python kbi_analysis.py -s [project_path] -p [pure_component_path]

Options include {known_options}
or their shorthand versions {shorthand_options}

--prj_path or -s:  
    path to project directory containing system folders
    default: current working directory

--pure_component_path or -p: 
    path to pure component directory containing systems of pure components
    default: pure_components, located in parent directory

--rdf_dir or -r: 
    name for rdf directory in each system
    default: rdf_files

--kbi_method or -k: 
    kbi method name
    default: adj
    options:
      - raw: no correction
      - adj: correcting for R != 1
      - kgv: dampening of the signal & correction for R != 1

--start_time or -b:
    time in ns to start averaging, enthalpy, box volume, and temperature
    default: 100

--end_time or -e:
    time in ns to end averaging, enthalpy, box volume, and temperature
    default: end of trajectory

--thermo_model or -m: 
    thermodynamic model name for LLE calculation
    default: quartic
    options: quartic (numerical approach), uniquac, unifac, nrtl, fh
      - nrtl and fh are only supported for binary systems  

--help or -h:
    displays this message.
''')

if '-h' in list(sys_arg.argv) or '--help' in list(sys_arg.argv):
  printhelp()
  sys_arg.exit()


current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# specify default values
prj_path = f"{current_dir}/"
pure_component_path = f"{parent_dir}/pure_components/"
rdf_dir = 'rdf_files'
kbi_method = 'adj'
start_time = 100
end_time = None
thermo_model = 'quartic'

# extract values from options
for a, arg in enumerate(sys_arg.argv[:-1]):
  if (arg == '--prj_path') or (arg == '-s'):
    prj_path = sys_arg.argv[a+1]
  if (arg == '--pure_component_path') or (arg == '-p'):
    pure_component_path = sys_arg.argv[a+1]
  if (arg == '--rdf_dir') or (arg == '-r'):
    rdf_dir = sys_arg.argv[a+1]
  if (arg == '--kbi_method') or (arg == '-k'):
    kbi_method = sys_arg.argv[a+1]
  if (arg == '--start_time') or (arg == '-b'):
    start_time = sys_arg.argv[a+1]
  if (arg == '--end_time') or (arg == '-e'):
    end_time = sys_arg.argv[a+1]
  if (arg == '--thermo_model') or (arg == '-m'):
    thermo_model = sys_arg.argv[a+1]

# initialize kbi object
kbi_obj = KBI(prj_path=prj_path, pure_component_path=pure_component_path, rdf_dir=rdf_dir, kbi_method=kbi_method, avg_start_time=start_time, avg_end_time=end_time)

# run kbi analysis
kbi_obj.kbi_analysis()

# create figures
kbi_plotter = KBIPlotter(kbi_obj)
kbi_plotter.make_figures()

# create thermodynamic model
# specific temperatures for temperature scaling; dT: step, Tmin: lower bounds, Tmax: upper bounds
tmodel = ThermoModel(model_name=thermo_model, KBIModel=kbi_obj, dT=1, Tmin=150, Tmax=kbi_obj.T_sim)
tmodel.temperature_scaling()

# create phase diagram figures
tmodel_plotter = PhaseDiagramPlotter(tmodel)
tmodel_plotter.make_figures()