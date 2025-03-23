import os
import sys
import warnings
import argparse

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from . import ThermoModel, KBI, KBIPlotter, PhaseDiagramPlotter

def main():
  parser = argparse.ArgumentParser(description='PICMOL: A Phase Instability Calculator for Molecular Design')

  parser.add_argument('--prj_path', default=os.getcwd(), help='Path to project directory containing system folders (default: current working directory)')
  parser.add_argument('--pure_component_path', default=os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'pure_components')), help='Path to pure component directory (default: parent_dir/pure_components)')
  parser.add_argument('--rdf_dir', type=str, default='rdf_files', help='Name for rdf directory in each system (default: rdf_files)')
  parser.add_argument('--kbi_dir', type=str, default='kbi_analysis', help='Name for directory for kbi analysis (default: kbi_analysis)')
  parser.add_argument('--kbi_method', type=str, default='adj', choices=['raw', 'adj', 'kgv'], help='KBI method name (default: adj)')
  parser.add_argument('--start_time', type=float, default=100, help='Time in ns to start averaging (default: 100)')
  parser.add_argument('--end_time', type=float, default=None, help='Time in ns to end averaging (default: end of trajectory)')
  parser.add_argument('--thermo_model', type=str, default='quartic', choices=['quartic', 'uniquac', 'unifac', 'nrtl', 'fh'], help='Thermodynamic model name for LLE calculation (default: quartic)')
  parser.add_argument('--Tmin', type=float, default=150, help='Minimum Temp (K) for temperature sacling (default: 150)')
  parser.add_argument('--Tmax', type=float, default=400, help='Maximum Temp (K) for temperature scaling (default: 400)')
  parser.add_argument('--dT', type=float, default=1, help='Temperature step (K) for temperature scaling (default: 1) ')


  args = parser.parse_args()

  # initialize kbi object
  print('intializing kbi object')
  kbi_obj = KBI(prj_path=args.prj_path, pure_component_path=args.pure_component_path, rdf_dir=args.rdf_dir, kbi_method=args.kbi_method, avg_start_time=args.start_time, avg_end_time=args.end_time, kbi_fig_dirname=args.kbi_dir)

  # run kbi analysis
  print('computing kbis')
  kbi_obj.kbi_analysis()

  # create figures
  print('creating figures')
  kbi_plotter = KBIPlotter(kbi_obj)
  kbi_plotter.make_figures()

  # create thermodynamic model
  print('initializing thermodynamic model')
  tmodel = ThermoModel(model_name=args.thermo_model, KBIModel=kbi_obj, dT=args.dT, Tmin=args.Tmin, Tmax=args.Tmax)

  print('performing temperature scaling')
  tmodel.temperature_scaling()

  # create phase diagram figures
  print('creating figures')
  tmodel_plotter = PhaseDiagramPlotter(tmodel)
  tmodel_plotter.make_figures()

if __name__ == "__main__":
  main()