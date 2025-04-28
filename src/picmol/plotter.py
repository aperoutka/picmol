from re import I
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from matplotlib.ticker import MultipleLocator
from pathlib import Path
plt.style.use(Path(__file__).parent / 'presentation.mplstyle')
import os, math, copy
import numpy as np 
import plotly.graph_objects as go
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import mpltern

from .thermo_model import ThermoModel
from .conversions import mol2vol
from .kbi import add_zeros


def get_cmap(xx, colormap: str = 'jet', levels: int = 40):
  c = xx
  norm = mpl.colors.Normalize(vmin=min(c), vmax=max(c))
  cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap, levels))
  cmap.set_array([])
  return c, cmap


class KBIPlotter:
  def __init__(self, model):
    """ visualizing kbi analysis """
    self.kbi_model = model
    self.num_comp = len(self.kbi_model.unique_mols)
    
  def make_figures(self, show_fig: bool = False):
    '''creates figures based on number of components in system'''
    # kbi integral as a function of r
    self.make_indiv_kbi_plots(show_fig=show_fig) # running kbi
    self.plot_kbi_inf(show_fig=show_fig) # kbi extrapolation to infinity (thermodynamic limit)

    for basis in ["mol", "vol"]:
      # kbi values & activities
      self.plot_composition_kbis(basis=basis, show_fig=show_fig)
      self.plot_dln_gammas(basis=basis, show_fig=show_fig)
      self.plot_ln_gammas(basis=basis, show_fig=show_fig)

      if self.num_comp == 2:
        # calculated excess parameters
        self.plot_binary_Gex(basis=basis, show_fig=show_fig)
        self.plot_binary_excess_contributions(basis=basis, show_fig=show_fig)
        # plot fits for thermodynamic interaction parameters
        self.plot_NRTL_IP_fit()
        self.plot_FH_chi_fit()
        self.plot_UNIQUAC_IP_fit()
        self.plot_quartic_fit()
        self.plot_binary_thermo_model_comparisons(show_fig=show_fig)

  def make_indiv_kbi_plots(self, show_fig: bool = False):
    ''' create plots for each system as a function of r '''
    # if kbi's not found, run kbi_analysis
    try:
      self.kbi_model.df_kbi
    except AttributeError: 
      self.kbi_model.kbi_analysis()

    for s, sys in enumerate(self.kbi_model.systems):
      fig, ax = plt.subplots(1, self.kbi_model.ij_combo, figsize=(12, 4), sharex=True)
      ij_combo = 0
      for i, mol_1 in enumerate(self.kbi_model._top_unique_mols):
        for j, mol_2 in enumerate(self.kbi_model._top_unique_mols):
          if i <= j:
            # get kbi's as a function of r
            df_kbi_sys = getattr(self.kbi_model, f"kbi_{s}")
            ax[ij_combo].plot(df_kbi_sys["r"], df_kbi_sys[f'G_{mol_1}_{mol_2}_cm3_mol'], c="tab:red", linestyle='solid', linewidth=4, alpha=0.7)
            # figure properties
            ax[ij_combo].set_xlim(0, 5)
            ax[ij_combo].set_xticks(ticks=np.arange(0,5.1,1.))
            ax[ij_combo].set_xlabel('$r$ [nm]')
            ax[ij_combo].set_title(f"{self.kbi_model.mol_name_dict[mol_1]}-{self.kbi_model.mol_name_dict[mol_2]}\n{self.kbi_model.df_comp[f'phi_{self.kbi_model.solute}'][s]:.2f} $\phi_{{{self.kbi_model.solute_name}}}$")
            ij_combo += 1
      ax[0].set_ylabel('$G_{ij}^R$ [cm$^3$ mol$^{-1}$]')
      plt.savefig(f'{self.kbi_model.kbi_indiv_fig_dir}{sys}_kbi.png')
      if show_fig:
        plt.show()
      else:
        plt.close()

  def plot_kbi_inf(self, show_fig: bool = False):
    ''' create kbi plots for extrapolating to the thermodynamic limit'''
    try:
      self.kbi_model.df_kbi
    except AttributeError: 
      self.kbi_model.kbi_analysis()

    for s, sys in enumerate(self.kbi_model.systems):
      fig, ax = plt.subplots(1, self.kbi_model.ij_combo, figsize=(12, 4), sharex=True)
      ij_combo = 0
      for i, mol_1 in enumerate(self.kbi_model._top_unique_mols):
        for j, mol_2 in enumerate(self.kbi_model._top_unique_mols):
          if i <= j:
            # get kbi's as a function of r
            df_kbi_sys = getattr(self.kbi_model, f"kbi_{s}")
            Gij_R = df_kbi_sys[f'G_{mol_1}_{mol_2}_cm3_mol']
            r = df_kbi_sys["r"]

            L = self.kbi_model.lamdba_values[sys][f'{mol_1}-{mol_2}']
            L_fit = self.kbi_model.lamdba_values_fit[sys][f'{mol_1}-{mol_2}']
            inf_coeffs = self.kbi_model.kbi_inf_fits[sys][f'{mol_1}-{mol_2}']
            Gij = inf_coeffs[0]

            ax[ij_combo].plot(L, L*Gij_R, c="dodgerblue", linestyle='solid', linewidth=2, alpha=0.5)
            ax[ij_combo].plot(L_fit, inf_coeffs(L_fit),c='k', alpha=0.9, ls='--', lw=3, label=f"$G_{{ij}}^{{\infty}}$: {Gij:.0f}")
            ax[ij_combo].legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
            # figure properties
            ax[ij_combo].set_xlim(L.min(), L.max())
            ax[ij_combo].set_xlabel('$\lambda$')
            ax[ij_combo].set_title(f"{self.kbi_model.mol_name_dict[mol_1]}-{self.kbi_model.mol_name_dict[mol_2]}\n{self.kbi_model.df_comp[f'phi_{self.kbi_model.solute}'][s]:.2f} $\phi_{{{self.kbi_model.solute_name}}}$")
            ij_combo += 1
      ax[0].set_ylabel('$\lambda$ $G_{ij}^R$ [cm$^3$ mol$^{-1}$]')
      plt.savefig(f'{self.kbi_model.kbi_indiv_fig_dir}{sys}_kbi_inf.png')
      if show_fig:
        plt.show()
      else:
        plt.close()


  def x_basis(self, basis):
    # get xlabel for figure and column label in df for mol frac / vol frac basis
    if basis.lower() == "mol":
      zplot = self.kbi_model.z
      xplot = self.kbi_model.z[:,self.kbi_model.solute_loc].flatten()
      x_lab = 'x'
    else:
      zplot = self.kbi_model.v
      xplot = self.kbi_model.v[:,self.kbi_model.solute_loc].flatten()
      x_lab = '\phi'
    return zplot, xplot, x_lab

  def plot_composition_kbis(self, basis, show_fig: bool = False):
    '''kbi integral values as a function of composition'''
    try:
      self.kbi_model.df_kbi
    except AttributeError: 
      self.kbi_model.kbi_analysis()

    _, _, x_lab = self.x_basis(basis)
    if basis.lower() == "mol":
      xplot = self.kbi_model._top_z[:,self.kbi_model._top_solute_loc].flatten()
    else:
      xplot = self.kbi_model._top_v[:,self.kbi_model._top_solute_loc].flatten()

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    colors = plt.cm.jet(np.linspace(0,1,len(self.kbi_model._top_unique_mols)+1))
    ij = 0
    for i, mol_1 in enumerate(self.kbi_model._top_unique_mols):
      for j, mol_2 in enumerate(self.kbi_model._top_unique_mols):
        if i <= j:
          ax.scatter(xplot, self.kbi_model.df_kbi[f'G_{mol_1}_{mol_2}_cm3_mol'], c=colors[ij], marker='s', linewidth=1.8, label=f'{self.kbi_model.mol_name_dict[mol_1]}-{self.kbi_model.mol_name_dict[mol_2]}')
          ij += 1

    ax.legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(ticks=np.arange(0,1.1,0.1))
    ax.set_xlabel(f'${x_lab}_{{{self.kbi_model.solute_name}}}$')
    ax.set_ylabel(f'$G_{{ij}}^{{\infty}}$ [cm$^3$ mol$^{{-1}}$]')
    plt.savefig(f'{self.kbi_model.kbi_method_dir}composition_KBI_{basis.lower()}frac_{self.kbi_model.kbi_method.lower()}.png')
    if show_fig:
      plt.show()
    else:
      plt.close()

  def plot_dln_gammas(self, basis, ylimits=[], show_fig: bool = False):
    '''derivative of log activity coefficients'''
    zplot, xplot, x_lab = self.x_basis(basis)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    colors = plt.cm.jet(np.linspace(0,1,len(self.kbi_model.unique_mols)+1))
    for i, mol in enumerate(self.kbi_model.unique_mols):
      if zplot.shape[1] > 2:
        ax.scatter(zplot[:,i], self.kbi_model.dlngamma_dxs[:,i], c=colors[i], linewidth=1.8, marker='s', label=self.kbi_model.mol_name_dict[mol])
        ax.set_xlabel(f'${x_lab}_i$')
      else:
        ax.scatter(zplot[:,self.kbi_model.solute_loc], self.kbi_model.dlngamma_dxs[:,i], c=colors[i], linewidth=1.8, marker='s', label=self.kbi_model.mol_name_dict[mol])
        ax.set_xlabel(f'${x_lab}_{{{self.kbi_model.solute_name}}}$')
    ax.legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    if len(ylimits) > 0:
      ax.set_ylim(ylimits)
    else:
      dg = self.kbi_model.dlngamma_dxs.max() - self.kbi_model.dlngamma_dxs.min()
      gamma_min = self.kbi_model.dlngamma_dxs.min() - 0.1*dg
      gamma_max = self.kbi_model.dlngamma_dxs.max() + 0.1*dg
      ymin = min([-0.05, gamma_min])
      ymax = max([0.05, gamma_max])
      ax.set_ylim(ymin, ymax)
    ax.set_xticks(ticks=np.arange(0,1.1,0.1))
    ax.set_ylabel('$\partial \ln(\gamma_{i})/\partial x_{i}$')
    plt.savefig(f'{self.kbi_model.kbi_method_dir}deriv_activity_coefs_{basis}frac_{self.kbi_model.kbi_method.lower()}.png')
    if show_fig:
      plt.show()
    else:
      plt.close()

  def plot_ln_gammas(self, basis, ylimits=[], show_fig: bool = False):
    '''log activity coefficients'''
    zplot, xplot, x_lab = self.x_basis(basis)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    colors = plt.cm.jet(np.linspace(0,1,len(self.kbi_model.unique_mols)+1))
    for i, mol in enumerate(self.kbi_model.unique_mols):
      if zplot.shape[1] > 2:
        ax.scatter(zplot[:,i], np.log(self.kbi_model.gammas[:,i]), c=colors[i], linewidth=1.8, marker='s', label=self.kbi_model.mol_name_dict[mol])
        ax.set_xlabel(f'${x_lab}_i$')
      else:
        ax.scatter(zplot[:,self.kbi_model.solute_loc], np.log(self.kbi_model.gammas[:,i]), c=colors[i], linewidth=1.8, marker='s', label=self.kbi_model.mol_name_dict[mol])
        ax.set_xlabel(f'${x_lab}_{{{self.kbi_model.solute_name}}}$')
    ax.legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    if len(ylimits) > 0:
      ax.set_ylim(ylimits)
    else:
      dg = np.log(self.kbi_model.gammas.max()) - np.log(self.kbi_model.gammas.min())
      gamma_min = np.log(self.kbi_model.gammas.min()) - 0.1*dg
      gamma_max = np.log(self.kbi_model.gammas.max()) + 0.1*dg
      ymin = min([-0.05, gamma_min])
      ymax = max([0.05, gamma_max])
      ax.set_ylim(ymin, ymax)
    ax.set_xticks(ticks=np.arange(0,1.1,0.1))
    ax.set_ylabel('$\ln \gamma_{i}$')
    plt.savefig(f'{self.kbi_model.kbi_method_dir}activity_coefs_{basis}frac_{self.kbi_model.kbi_method.lower()}.png')
    if show_fig:
      plt.show()
    else:
      plt.close()

  def xplot0(self, basis):
    zplot, xplot, _ = self.x_basis(basis)
    xplot0 = np.zeros(len(xplot)+2)
    xplot0[1:-1] = xplot
    if xplot0[-1] > xplot0[1]:
      xplot0[-1] = 1
    else:
      xplot0[0] = 1
    return xplot0
  
  def plot_binary_Gex(self, basis, show_fig: bool = False):

    zplot, xplot, x_lab = self.x_basis(basis)
    xplot0 = self.xplot0(basis)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(xplot0, add_zeros(self.kbi_model.G_ex), c='m', linewidth=1.8, marker='s')
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(ticks=np.arange(0,1.1,0.1))
    ax.set_xlabel(f'${x_lab}_{{{self.kbi_model.solute_name}}}$')
    ax.set_ylabel('$G^E$ $[kJ$ $mol^{-1}]$')
    plt.savefig(f'{self.kbi_model.kbi_method_dir}gibbs_excess_energy_{basis}frac_{self.kbi_model.kbi_method.lower()}.png')
    if show_fig:
      plt.show()
    else:
      plt.close()

  def plot_binary_excess_contributions(self, basis, show_fig: bool = False):
    zplot, xplot, x_lab = self.x_basis(basis)
    xplot0 = self.xplot0(basis)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(xplot0, add_zeros(self.kbi_model.G_ex), c='violet', linewidth=1.8, marker='s', label="$G^E$")
    ax.scatter(xplot0, add_zeros(self.kbi_model.Hmix), c='mediumblue', linewidth=1.8, marker='o', label="$\Delta$ $H_{mix}$")
    ax.scatter(xplot0, -self.kbi_model.T_sim * add_zeros(self.kbi_model.S_ex), c='limegreen', linewidth=1.8, marker='^', label="$-TS^E$")
    ax.legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(ticks=np.arange(0,1.1,0.1))
    ax.set_xlabel(f'${x_lab}_{{{self.kbi_model.solute_name}}}$')
    ax.set_ylabel('Excess Properties $[kJ$ $mol^{-1}]$')
    plt.savefig(f'{self.kbi_model.kbi_method_dir}gibbs_excess_properties_{basis}frac_{self.kbi_model.kbi_method.lower()}.png')
    if show_fig:
      plt.show()
    else:
      plt.close()

  def plot_binary_thermo_model_comparisons(self, show_fig: bool = False):
    """compare UNIQUAC, UNIFAC, QuarticModel"""
    fig, ax = plt.subplots(1, 3, figsize=(12,3.75), sharex=True)
    xplot = self.kbi_model.z_plot[:,self.kbi_model.solute_loc]
    
    ax[0].scatter(self.kbi_model.z[:,self.kbi_model.solute_loc], self.kbi_model.Hmix, c='k', zorder=10)
    ax[0].plot(xplot, self.kbi_model.quartic_Hmix, c='k', ls='solid')
    ax[0].plot(xplot, self.kbi_model.uniquac_Hmix, c='dodgerblue', ls='dashed')
    ax[0].plot(xplot, self.kbi_model.unifac_Hmix, c='limegreen', ls='dotted')

    ax[1].scatter(self.kbi_model.z[:,self.kbi_model.solute_loc], self.kbi_model.nTdSmix, c='k', label='KB + MD', zorder=10)
    ax[1].plot(xplot, self.kbi_model.quartic_Smix, c='k', ls='solid', label='Fit')
    ax[1].plot(xplot, self.kbi_model.uniquac_Smix, c='dodgerblue', ls='dashed', label='uniquac')
    ax[1].plot(xplot, self.kbi_model.unifac_Smix, c='limegreen', ls='dotted', label='unifac')

    uniq_G = self.kbi_model.uniquac_Hmix + self.kbi_model.uniquac_Smix
    unif_G = self.kbi_model.unifac_Hmix + self.kbi_model.unifac_Smix
    quar_G = self.kbi_model.quartic_Hmix + self.kbi_model.quartic_Smix

    ax[2].scatter(self.kbi_model.z[:,self.kbi_model.solute_loc], self.kbi_model.G_mix_xv, c='k', zorder=10)
    ax[2].plot(xplot, quar_G, c='k', ls='solid')
    ax[2].plot(xplot, uniq_G, c='dodgerblue', ls='dashed')
    ax[2].plot(xplot, unif_G, c='limegreen', ls='dotted')

    ax[0].set_xlim(-0.05,1.05)
    for i in range(3):
      ax[i].set_xlabel(f'$x_{{{self.kbi_model.solute_name}}}$')
    ax[0].set_ylabel(f'$\Delta H_{{mix}}$ $[kJ$ $mol^{{-1}}]$')
    ax[1].set_ylabel(f'$-T\Delta S_{{mix}}$ $[kJ$ $mol^{{-1}}]$')
    ax[2].set_ylabel(f'$\Delta G_{{mix}}$ $[kJ$ $mol^{{-1}}]$')

    ax[1].legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    plt.savefig(f'{self.kbi_model.kbi_method_dir}thermo_model_comparisons_{self.kbi_model.kbi_method.lower()}.png')
    if show_fig:
      plt.show()
    else:
      plt.close()


  def plot_UNIQUAC_IP_fit(self):
    xplot0 = self.xplot0("mol")

    fig, ax = plt.subplots()
    ax.scatter(xplot0, add_zeros(self.kbi_model.G_mix_xv), marker='o', linewidth=1.8, color='k', label='Simulated', zorder=10)
    uniq_G = self.kbi_model.uniquac_Hmix + self.kbi_model.uniquac_Smix
    ax.plot(self.kbi_model.z_plot[:,self.kbi_model.solute_loc], uniq_G, linewidth=2.5, color='tab:red', label='UNIQUAC')      
    ax.set_xlim(-0.05,1.05)
    ax.set_xlabel(f'$x_{{{self.kbi_model.solute_name}}}$')
    ax.set_ylabel(f'$\Delta G_{{mix}}$ $[kJ$ $mol^{{-1}}]$')
    ax.legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    plt.savefig(f'{self.kbi_model.kbi_method_dir}UNIQUAC_fit_molfrac_{self.kbi_model.kbi_method.lower()}.png')
    plt.close()

  def plot_NRTL_IP_fit(self):
    # if not a binary system skip the function
    if len(self.kbi_model.unique_mols) != 2:
      return

    try:
      self.kbi_model.nrtl_taus
    except:
      self.kbi_model.fit_binary_NRTL_IP()

    tau12, tau21 = list(self.kbi_model.nrtl_taus.values())

    def NRTL_GE(z, tau12, tau21):
      alpha = 0.2 # randomness factor == constant
      G12 = np.exp(-alpha*tau12/(self.kbi_model.Rc*self.kbi_model.T_sim))
      G21 = np.exp(-alpha*tau21/(self.kbi_model.Rc*self.kbi_model.T_sim))
      x1 = z[:,0]
      x2 = z[:,1]
      G_ex = -self.kbi_model.Rc * self.kbi_model.T_sim * (x1 * x2 * (tau21 * G21/(x1 + x2 * G21) + tau12 * G12 / (x2 + x1 * G12))) 
      G_id = self.kbi_model.Rc * self.kbi_model.T_sim * (x1 * np.log(x1) + x2 * np.log(x2))
      return G_ex + G_id
    
    Gmix_fit0 = NRTL_GE(self.kbi_model.z_plot, tau12, tau21)
    Gmix_fit0 = np.nan_to_num(Gmix_fit0, nan=0)

    xplot0 = self.xplot0("mol")

    fig, ax = plt.subplots()
    ax.scatter(xplot0, add_zeros(self.kbi_model.G_mix_xv), marker='o', linewidth=1.8, color='k', label='Simulated', zorder=10)
    ax.plot(self.kbi_model.z_plot[:,self.kbi_model.solute_loc], Gmix_fit0, linewidth=2.5, color='tab:red', label='NRTL')      
    ax.set_xlim(-0.05,1.05)
    ax.set_xlabel(f'$x_{{{self.kbi_model.solute_name}}}$')
    ax.set_ylabel(f'$\Delta G_{{mix}}$ $[kJ$ $mol^{{-1}}]$')
    ax.legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    plt.savefig(f'{self.kbi_model.kbi_method_dir}NRTL_fit_molfrac_{self.kbi_model.kbi_method.lower()}.png')
    plt.close()


  def plot_FH_chi_fit(self):
    try:
      self.kbi_model.fh_chi
    except:
      self.kbi_model.fit_FH_chi()

    pplot = np.zeros(len(self.kbi_model.fh_phi)+2)
    pplot[1:-1] = self.kbi_model.fh_phi
    pplot[-1] = 1

    Gmix0 = np.zeros(len(self.kbi_model.G_mix_xv)+2)
    Gmix0[1:-1] = self.kbi_model.G_mix_xv
    fh_gmix0 = np.zeros(len(self.kbi_model.fh_Gmix)+2)
    fh_gmix0[1:-1] = self.kbi_model.fh_Gmix

    fig, ax = plt.subplots()
    ax.scatter(pplot, Gmix0, color='k', marker='o', linewidth=1.8, label="Simulated", zorder=10)
    ax.plot(pplot, fh_gmix0, color='tab:red', linewidth=2.5, label='FH')
    ax.legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    ax.set_xlabel(f'$\phi_{{{self.kbi_model.solute_name}}}$')
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(ticks=np.arange(0,1.1,0.2))
    ax.set_ylabel(f'$\Delta G_{{mix}}$ $[kJ$ $mol^{{-1}}]$')
    plt.savefig(f'{self.kbi_model.kbi_method_dir}FH_fit_volfrac_{self.kbi_model.kbi_method.lower()}.png')
    plt.close()

  def plot_quartic_fit(self):
    xplot0 = self.xplot0("mol")

    fig, ax = plt.subplots()
    ax.scatter(xplot0, add_zeros(self.kbi_model.G_mix_xv), marker='o', linewidth=1.8, color='k', label='Simulated', zorder=10)
    ax.plot(self.kbi_model.z_plot[:,self.kbi_model.solute_loc], self.kbi_model.quartic_Smix + self.kbi_model.quartic_Hmix, linewidth=2.5, color='tab:red', label='Quartic')      
    ax.set_xlim(-0.05,1.05)
    ax.set_xlabel(f'$x_{{{self.kbi_model.solute_name}}}$')
    ax.set_ylabel(f'$\Delta G_{{mix}}$ $[kJ$ $mol^{{-1}}]$')
    ax.legend(fontsize=11, labelspacing=0.5, frameon=True, edgecolor='k', framealpha=0.5)
    plt.savefig(f'{self.kbi_model.kbi_method_dir}QUARTIC_fit_molfrac_{self.kbi_model.kbi_method.lower()}.png')
    plt.close()





class PhaseDiagramPlotter:
  def __init__(self, model: ThermoModel):
    """ for visualizing phase behavior """
    self.thermo_model = model

  def make_figures(self, T=300, colormap='jet', num_contours=40):
    if self.thermo_model.z.shape[1] == 2:
      for basis in ["mol", "vol"]:
        self.binary_gmix(basis=basis)
        self.binary_gmix_selectpts(basis=basis)
        self.binary_phase_diagram_Gmix_heatmap(basis=basis, num_contours=num_contours)
        self.binary_phase_diagram(basis=basis)
        self.binary_phase_diagram_I0_heatmap(basis=basis, num_contours=num_contours)
        self.binary_phase_diagram_I0_heatmap_widomline(basis=basis, num_contours=num_contours)
        self.binary_phase_diagram_widomline(basis=basis)

    elif self.thermo_model.z.shape[1] == 3:
      self.ternary_GM(T, plot_spbi=False, colormap=colormap, num_contours=num_contours)
      self.ternary_GM(T, plot_spbi=True, colormap=colormap, num_contours=num_contours)
      self.ternary_Io(T, plot_spbi=False, colormap=colormap, num_contours=num_contours)
      self.ternary_Io(T, plot_spbi=True, colormap=colormap, num_contours=num_contours)
      self.ternary_binodals_fTemp(colormap=colormap)
    
    elif self.thermo_model.z.shape[1] > 3:
      print("plotter functions only supported for binary and ternary systems")
  

  def x_basis(self, basis):
    if basis == "mol":
      x_val = self.thermo_model.z[:,self.thermo_model.kbi_model.solute_loc].flatten()
      x_lab = 'x'
      sp = self.thermo_model.x_sp
      bi = self.thermo_model.x_bi
    else:
      x_val = self.thermo_model.v[:,self.thermo_model.kbi_model.solute_loc].flatten()
      x_lab = '\phi'
      sp = self.thermo_model.v_sp
      bi = self.thermo_model.v_bi
    return x_val, x_lab, sp, bi
   
  def binary_gmix(self, basis: str, show_fig: bool = False):
    '''get mixing free energy with binodal & spinodal points'''
    c, cmap = get_cmap(self.thermo_model.T_values)
    x_val, x_lab, sp, bi = self.x_basis(basis)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for t, T in enumerate(self.thermo_model.T_values):
      if T % 10 == 0:
        # add mixing free energy to plot
        ax.plot(x_val, self.thermo_model.GM[t], lw=2, c=cmap.to_rgba(c[t]))
        # add spinodals to plot
        ax.plot(sp[t], self.thermo_model.GM_sp[t], marker='o', color='k', linestyle='', fillstyle='none', linewidth=1.5, zorder=len(self.thermo_model.T_values)+1)
        # add binodals to plot
        ax.plot(bi[t], self.thermo_model.GM_bi[t], marker='o', color='k', linestyle='', fillstyle='full', linewidth=1.5, zorder=len(self.thermo_model.T_values)+1)
    fig.colorbar(cmap, ax=ax, orientation='vertical', pad=0.02, label='Temperature [K]')
    ax.set_xlabel(f"${x_lab}_{{{self.thermo_model.kbi_model.solute_name}}}$")
    ax.set_xlim(-0.05,1.05)
    ax.set_xticks(ticks=np.arange(0,1.01,0.2))
    ax.set_ylabel(f"$\Delta G_{{mix}}$ $[kJ$ $mol^{{-1}}$]")
    if self.thermo_model.save_dir is not None:
      plt.savefig(f"{self.thermo_model.save_dir}/{self.thermo_model.model_name}_Gmix_bin_spin_{basis}frac.png")
    if show_fig:
      plt.show()
    else:
      plt.close()

  def binary_gmix_selectpts(self, basis: str, show_fig: bool = False):
    ''' get select mixing free energy lines '''
    c, cmap = get_cmap(self.thermo_model.T_values)
    x_val, x_lab, sp, bi = self.x_basis(basis)
    # get temperature just above phase spliting, 5 degree intervals
    Tcs = 5 * (self.thermo_model.Tc // 5)
    step = 20 # Temperature differences to plot
    # temperatures
    Ts = [Tcs + i*step for i in [1, 0, -1, -2, -3]]
    Ts_inds = [t for t, T in enumerate(self.thermo_model.T_values) if T in Ts]
    # colors
    cs = plt.cm.jet(np.linspace(0,1,len(Ts)+1))

    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    c_ind = 0
    for t in Ts_inds:
      T = self.thermo_model.T_values[t]
      ax.plot(x_val, self.thermo_model.GM[t], lw=2, c=cs[c_ind], label=f'{T:.0f} K')
      # add spinodals
      ax.plot(sp[t], self.thermo_model.GM_sp[t], marker='o', color='k', linestyle='', fillstyle='none', linewidth=1.5, zorder=len(self.thermo_model.T_values)+1)
      # add binodals
      ax.plot(bi[t], self.thermo_model.GM_bi[t], marker='o', color='k', linestyle='', fillstyle='full', linewidth=1.5, zorder=len(self.thermo_model.T_values)+1)
      c_ind += 1
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel(f"${x_lab}_{{{self.thermo_model.kbi_model.solute_name}}}$")
    ax.set_xlim(-0.05,1.05)
    ax.set_xticks(ticks=np.arange(0,1.01,0.2))
    ax.set_ylabel(f"$\Delta G_{{mix}}$ $[kJ$ $mol^{{-1}}$]")
    if self.thermo_model.save_dir is not None:
      plt.savefig(f"{self.thermo_model.save_dir}/{self.thermo_model.model_name}_Gmix_binodal_spinodal_selectpts_{basis}frac.png")
    if show_fig == True:
      plt.show()
    else:
      plt.close()


  def binary_phase_diagram_Gmix_heatmap(self, basis: str, num_contours=40, show_fig: bool = False):

    x_val, x_lab, sp, bi = self.x_basis(basis)

    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.grid(False)
    ax.tick_params(bottom=False, top=False, left=False, right=False)

    cs = ax.contourf(x_val, self.thermo_model.T_values, self.thermo_model.GM, cmap=plt.cm.jet, levels=num_contours)
    fig.colorbar(cs, ax=ax, orientation='vertical', pad=0.01, format=mpl.ticker.FormatStrFormatter('%.2f'), label='$\Delta G_{{mix}}$ $[kJ$ $mol^{{-1}}]$')

    ax.plot(sp[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(sp[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)

    ax.set_xlim(0, 1.)
    ax.set_xticks(ticks=np.arange(0,1.01,0.2))
    ax.set_xlabel(f'${x_lab}_{{{self.thermo_model.kbi_model.solute_name}}}$')  
    ax.set_ylabel('Temperature [K]')
    if self.thermo_model.save_dir is not None:
      plt.savefig(f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_phase_diagram_Gmix_heatmap_{basis}frac.png')
    if show_fig == True:
      plt.show()
    else:
      plt.close()

  def binary_phase_diagram_I0_heatmap(self, ymin: float = 0.01, ymax: float = 1., basis: str = 'vol', num_contours=40, show_fig: bool = False):
    
    x_val, x_lab, sp, bi = self.x_basis(basis)

    fig, ax = plt.subplots(1, 1, figsize=(5.5,4))
    ax.grid(False)
    ax.tick_params(bottom=False, top=False, left=False, right=False)

    I0_plot = np.zeros(self.thermo_model.I0_arr.shape)
    for t, T in enumerate(self.thermo_model.T_values):
      I0_arr_T = self.thermo_model.I0_arr[t]
      min_mask = (I0_arr_T >= ymin)
      max_mask = (I0_arr_T <= ymax)
      I0_plot[t] = I0_arr_T.copy()
      I0_plot[t][~min_mask] = ymin
      I0_plot[t][~max_mask] = ymax
      if np.all(~np.isnan(sp[t])): 
        sp_mask = (x_val > sp[t,0]) & (x_val < sp[t,1])
        I0_plot[t][sp_mask] = np.nan
    
    log_base = 10
    ymin_exp = np.round(math.log(ymin, log_base), 0)
    ymax_exp = np.round(math.log(ymax, log_base), 0)
    levels = np.logspace(ymin_exp, ymax_exp, num_contours)
    ticks = np.logspace(ymin_exp, ymax_exp, int(ymax_exp-ymin_exp+1))

    # create heatmap
    X, Y = np.meshgrid(x_val, self.thermo_model.T_values)
    cs = ax.contourf(X, Y, I0_plot, cmap=plt.cm.jet, norm=mpl.colors.LogNorm(vmin=ymin, vmax=ymax), levels=levels)
    cbar = fig.colorbar(cs, ax=ax, ticks=ticks, pad=0.01, orientation='vertical')
    cbar.ax.set_title('$I_o$ (cm$^{-1}$)', fontsize=12)
    cbar.ax.minorticks_on()

    # add spinodals and binodals
    ax.plot(sp[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(sp[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)

    ax.set_xlim(0, 1.)
    ax.set_xticks(ticks=np.arange(0,1.01,0.2))
    ax.set_xlabel(f'${x_lab}_{{{self.thermo_model.kbi_model.solute_name}}}$')  
    ax.set_ylabel('Temperature [K]')
    if self.thermo_model.save_dir is not None:
      plt.savefig(f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_phase_diagram_I0_heatmap_{basis}frac.png')
    if show_fig == True:
      plt.show()
    else:
      plt.close()

  def widom(self, basis):
    ''' return widom line for I0 max '''
    def moving_avg(data, window_size):
      '''get moving average to smooth the I0 max line'''
      moving_averages = []
      for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
      return moving_averages
    
    if basis == "mol":
      I0_max = copy.deepcopy(self.thermo_model.x_I0_max)
      for i in range(I0_max.shape[1]):
        I0_max[1:-1,i] = moving_avg(I0_max[:,i], 3)
      return I0_max
    else:
      I0_max = copy.deepcopy(self.thermo_model.v_I0_max)
      for i in range(I0_max.shape[1]):
        I0_max[1:-1,i] = moving_avg(I0_max[:,i], 3)
      return I0_max


  def binary_phase_diagram_I0_heatmap_widomline(self, ymin: float = 0.01, ymax: float = 1., basis: str = 'vol', num_contours=40, show_fig: bool = False):
    
    x_val, x_lab, sp, bi = self.x_basis(basis)

    fig, ax = plt.subplots(1, 1, figsize=(5.5,4))
    ax.grid(False)
    ax.tick_params(bottom=False, top=False, left=False, right=False)

    I0_plot = np.zeros(self.thermo_model.I0_arr.shape)
    for t, T in enumerate(self.thermo_model.T_values):
      I0_arr_T = self.thermo_model.I0_arr[t]
      min_mask = (I0_arr_T >= ymin)
      max_mask = (I0_arr_T <= ymax)
      I0_plot[t] = I0_arr_T.copy()
      I0_plot[t][~min_mask] = ymin
      I0_plot[t][~max_mask] = ymax
      if np.all(~np.isnan(sp[t])): 
        sp_mask = (x_val > sp[t,0]) & (x_val < sp[t,1])
        I0_plot[t][sp_mask] = np.nan
    
    log_base = 10
    ymin_exp = np.round(math.log(ymin, log_base), 0)
    ymax_exp = np.round(math.log(ymax, log_base), 0)
    levels = np.logspace(ymin_exp, ymax_exp, num_contours)
    ticks = np.logspace(ymin_exp, ymax_exp, int(ymax_exp-ymin_exp+1))

    # create heatmap
    X, Y = np.meshgrid(x_val, self.thermo_model.T_values)
    cs = ax.contourf(X, Y, I0_plot, cmap=plt.cm.jet, norm=mpl.colors.LogNorm(vmin=ymin, vmax=ymax), levels=levels)
    cbar = fig.colorbar(cs, ax=ax, ticks=ticks, pad=0.01, orientation='vertical')
    cbar.ax.set_title('$I_o$ (cm$^{-1}$)', fontsize=12)
    cbar.ax.minorticks_on()

    # add spinodals and binodals
    ax.plot(sp[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(sp[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)

    # add widom line
    ax.plot(self.widom(basis)[:,self.thermo_model.kbi_model.solute_loc], self.thermo_model.I0_max['T'], c='k', linestyle='solid', lw=2, zorder=len(self.thermo_model.T_values)+1)

    ax.set_xlim(0, 1.)
    ax.set_xticks(ticks=np.arange(0,1.01,0.2))
    ax.set_xlabel(f'${x_lab}_{{{self.thermo_model.kbi_model.solute_name}}}$')  
    ax.set_ylabel('Temperature [K]')
    if self.thermo_model.save_dir is not None:
      plt.savefig(f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_phase_diagram_I0_heatmap_widomline_{basis}frac.png')
    if show_fig == True:
      plt.show()
    else:
      plt.close()


  def xc(self, basis):
    ''' return critical point mol or vol frac '''
    if basis == "mol":
      return self.thermo_model.xc
    else:
      return self.thermo_model.phic

  def binary_phase_diagram(self, plot_Tmin=0, plot_Tmax=500, basis: str = 'vol', color='mediumblue', show_fig: bool = False):

    x_val, x_lab, sp, bi = self.x_basis(basis)

    fig, ax = plt.subplots(1, 1, figsize=(4.5,4))
    # add spinodals and binodals
    ax.plot(sp[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(sp[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)

    legend_label = f"$T_c = {self.thermo_model.Tc:.0f}$ K\n${x_lab}_c = {self.xc(basis):.3f}$"
    ax.plot(self.xc(basis), self.thermo_model.Tc, color='none', marker='o', linestyle='', fillstyle='none', label=legend_label)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(ticks=np.arange(0,1.1,0.2))

    if plot_Tmin != 0 and plot_Tmax != 500:
      ax.set_ylim(plot_Tmin, plot_Tmax)
    else:
      ax.set_ylim(min(self.thermo_model.T_values), max(self.thermo_model.T_values))

    ax.set_xlabel(f'${x_lab}_{{{self.thermo_model.kbi_model.solute_name}}}$')  
    ax.set_ylabel('Temperature [K]')
    if self.thermo_model.save_dir is not None:
      plt.savefig(f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_phase_diagram_{basis}frac.png')
    if show_fig == True:
      plt.show()
    else:
      plt.close()

  def binary_phase_diagram_widomline(self, plot_Tmin=0, plot_Tmax=500, basis: str = 'vol', color='mediumblue', show_fig: bool = False):

    x_val, x_lab, sp, bi = self.x_basis(basis)

    fig, ax = plt.subplots(1, 1, figsize=(4.5,4))
    # add spinodals and binodals
    ax.plot(sp[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(sp[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,0], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)
    ax.plot(bi[:,1], self.thermo_model.T_values, c='k', linestyle='solid', linewidth=2, zorder=len(self.thermo_model.T_values)+1)

    # add widom line
    ax.plot(self.widom(basis)[:,self.thermo_model.kbi_model.solute_loc], self.thermo_model.I0_max['T'], c='k', linestyle='solid', lw=2, zorder=len(self.thermo_model.T_values)+1)

    legend_label = f"$T_c = {self.thermo_model.Tc:.0f}$ K\n${x_lab}_c = {self.xc(basis):.3f}$"
    ax.plot(self.xc(basis), self.thermo_model.Tc, color='none', marker='o', linestyle='', fillstyle='none', label=legend_label)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(ticks=np.arange(0,1.1,0.2))

    if plot_Tmin != 0 and plot_Tmax != 500:
      ax.set_ylim(plot_Tmin, plot_Tmax)
    else:
      ax.set_ylim(min(self.thermo_model.T_values), max(self.thermo_model.T_values))

    ax.set_xlabel(f'${x_lab}_{{{self.thermo_model.kbi_model.solute_name}}}$')  
    ax.set_ylabel('Temperature [K]')
    if self.thermo_model.save_dir is not None:
      plt.savefig(f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_phase_diagram_widomline_{basis}frac.png')
    if show_fig == True:
      plt.show()
    else:
      plt.close()


#### ternary plots ####

  def ternary_GM(self, T, plot_spbi: bool = False, num_contours: int = 20, colormap: str = 'jet', show_fig: bool = False):
    ''' mixing free energy at T '''

    xtext, ytext, ztext = self.thermo_model.mol_name

    T_ind = np.abs(self.thermo_model.T_values - T).argmin()
    T_plot = self.thermo_model.T_values[T_ind]

    GM_arr = self.thermo_model.GM[T_ind]
    a, b, c = self.thermo_model.z[:,0], self.thermo_model.z[:,1], self.thermo_model.z[:,2]

    valid_mask = (a >= 0) & (b >= 0) & (c >= 0) & ~np.isnan(GM_arr) & ~np.isinf(GM_arr)
    a = a[valid_mask]
    b = b[valid_mask]
    c = c[valid_mask]
    values = GM_arr[valid_mask]

    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': 'ternary'})

    tp = ax.tricontourf(a, b, c, values, cmap=colormap, alpha=1, aspect=25, edgecolors='none', levels=num_contours)
    cbar = fig.colorbar(tp, ax=ax, aspect=25, label='$\Delta G_{mix}$ [kJ mol$^{-1}$]')

    if plot_spbi:
      figname = f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_GM_spbi_at_{T_plot}K.png'
      if self.thermo_model.model_name != 'unifac':
        sp_arr = np.array(self.thermo_model.x_sp[T_ind])
        ax.plot(sp_arr[:,0], sp_arr[:,1], sp_arr[:,2], color='k', linestyle='', marker='o', markersize=4, fillstyle='none')

      bi_arr = self.thermo_model.x_bi[T_ind]
      ax.plot(bi_arr[:,0], bi_arr[:,1], bi_arr[:,2], color='k', linestyle='', marker='o', markersize=4, fillstyle='full')
    else:
      figname = f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_GM_at_{T_plot}K.png'

    ax.set_tlabel(xtext)
    ax.set_llabel(ytext)
    ax.set_rlabel(ztext)

    ax.set_title(f'Mixing Free Energy at {int(T_plot)} K')

    # Add grid lines on top
    ax.grid(True, which='major', linestyle='-', linewidth=1, color='k')

    ax.taxis.set_major_locator(MultipleLocator(0.10))
    ax.laxis.set_major_locator(MultipleLocator(0.10))
    ax.raxis.set_major_locator(MultipleLocator(0.10))

    if self.thermo_model.save_dir is not None:
      plt.savefig(figname)
    if show_fig:
      plt.show()
    else:
      plt.close()


  def ternary_Io(self, T, plot_spbi: bool = False, num_contours: int = 20, colormap: str = 'jet', show_fig: bool = False):
    ''' mixing free energy at T '''

    xtext, ytext, ztext = self.thermo_model.mol_name

    T_ind = np.abs(self.thermo_model.T_values - T).argmin()
    T_plot = self.thermo_model.T_values[T_ind]

    try:
      self.thermo_model.I0_arr
    except:
      self.thermo_model.calculate_saxs_Io()

    Io_arr = self.thermo_model.I0_arr[T_ind]
    a, b, c = self.thermo_model.z[:,0], self.thermo_model.z[:,1], self.thermo_model.z[:,2]

    Io_arr[Io_arr <= 0.001] = 0.001
    values = Io_arr

    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': 'ternary'})

    log_base = 10
    ymin = 0.01
    ymax = 10
    ymin_exp = np.round(math.log(ymin, log_base), 0)
    ymax_exp = np.round(math.log(ymax, log_base), 0)
    levels = np.logspace(ymin_exp, ymax_exp, num_contours)
    ticks = np.logspace(ymin_exp, ymax_exp, int(ymax_exp-ymin_exp+1))

    tp = ax.tricontourf(a, b, c, values, cmap=colormap, 
                      alpha=1, aspect=25, edgecolors='none', levels=levels,
                      norm=mpl.colors.LogNorm(vmin=ymin), vmax=ymax)
    cbar = fig.colorbar(tp, ax=ax, ticks=ticks, pad=0.01, aspect=25, label='$I_o$ [cm$^{-1}$]')
    cbar.ax.minorticks_on()

    if plot_spbi:
      figname = f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_Io_spbi_at_{T_plot}K.png'
      if self.thermo_model.model_name != 'unifac':
        sp_arr = np.array(self.thermo_model.x_sp[T_ind])
        ax.plot(sp_arr[:,0], sp_arr[:,1], sp_arr[:,2], color='k', linestyle='', marker='o', markersize=4, fillstyle='none')

      bi_arr = self.thermo_model.x_bi[T_ind]
      ax.plot(bi_arr[:,0], bi_arr[:,1], bi_arr[:,2], color='k', linestyle='', marker='o', markersize=4, fillstyle='full')
    else:
      figname = f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_Io_at_{T_plot}K.png'


    ax.set_tlabel(xtext)
    ax.set_llabel(ytext)
    ax.set_rlabel(ztext)

    ax.set_title(f'SAXS $I_o$ at {int(T_plot)} K')

    # Add grid lines on top
    ax.grid(True, which='major', linestyle='-', linewidth=1, color='k')

    ax.taxis.set_major_locator(MultipleLocator(0.10))
    ax.laxis.set_major_locator(MultipleLocator(0.10))
    ax.raxis.set_major_locator(MultipleLocator(0.10))

    if self.thermo_model.save_dir is not None:
      plt.savefig(figname)
    if show_fig:
      plt.show()
    else:
      plt.close()


  def ternary_binodals_fTemp(self, colormap: str = 'jet', show_fig: bool = False):
    '''create ternary figure with heatmap of binodals as a function of temperature'''

    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': 'ternary'})

    cs, cmap = get_cmap(self.thermo_model.T_values, colormap=colormap)
    for a, arr in enumerate(self.thermo_model.x_bi):
      ax.plot(arr[:,0], arr[:,1], arr[:,2], linestyle='', marker='o', markersize=4, fillstyle='full', color=cmap.to_rgba(cs[a]), zorder=len(self.thermo_model.T_values)-a)
    fig.colorbar(cmap, ax=ax, aspect=25, label='Temperature [K]')

    xtext, ytext, ztext = self.thermo_model.mol_name
    ax.set_tlabel(xtext)
    ax.set_llabel(ytext)
    ax.set_rlabel(ztext)

    ax.set_title(f'Coexistence Temperature Dependence')

    # Add grid lines on top
    ax.grid(True, which='major', linestyle='-', linewidth=1, color='k')

    ax.taxis.set_major_locator(MultipleLocator(0.10))
    ax.laxis.set_major_locator(MultipleLocator(0.10))
    ax.raxis.set_major_locator(MultipleLocator(0.10))

    if self.thermo_model.save_dir is not None:
      plt.savefig(f'{self.thermo_model.save_dir}/{self.thermo_model.model_name}_binodals_temperature_dependence.png')
    if show_fig:
      plt.show()
    else:
      plt.close()


  def ternary_plotly_GM_3D(self, colormap: str = 'jet', show_fig: bool = False):
    ''' create 3D ternary plots of mixing free energy '''

    # Verify the shapes
    num_layers = len(self.thermo_model.T_values)

    # Flatten the ternary coordinates
    A = self.thermo_model.z[:,0]
    B = self.thermo_model.z[:,1]
    C = self.thermo_model.z[:,2]

    # Convert ternary coordinates to 2D Cartesian coordinates
    def ternary_to_cartesian(a, b, c):
      x = 0.5 * (2 * b + c)  # Simplified, since a + b + c = 1
      y = (np.sqrt(3) / 2) * c
      return x, y

    x_coords, y_coords = ternary_to_cartesian(A, B, C)

    # Prepare a mask for valid ternary coordinates
    tolerance = 1e-6
    sum_ABC = A + B + C
    valid_coords_mask = (
        (A >= 0) & (B >= 0) & (C >= 0) &
        np.isclose(sum_ABC, 1.0, atol=tolerance)
    )

    # Initialize lists to collect z-values across all layers
    all_z_values = []

    # Initialize the list of frames
    frames = []

    # Iterate over each GM layer to create frames
    for layer in range(num_layers):
        # Flatten the GM layer data
        GM_layer_flat = self.thermo_model.GM[layer].flatten()

        # Create a combined mask for valid coordinates and GM values
        GM_valid_mask = ~np.isnan(GM_layer_flat)
        combined_mask = valid_coords_mask & GM_valid_mask

        # Apply the combined mask
        x_layer = x_coords[combined_mask]
        y_layer = y_coords[combined_mask]
        z_layer = GM_layer_flat[combined_mask]

        # Collect z-values for global min and max
        all_z_values.extend(z_layer)

    # Calculate global z-axis limits
    if all_z_values:
        all_z_values = np.array(all_z_values).flatten()
        # remove nan values
        all_z_values = all_z_values[~np.isnan(all_z_values)]
        # remove inf values
        all_z_values = all_z_values[~np.isinf(all_z_values)]

        z_min = np.min(all_z_values)
        z_max = np.max(all_z_values)
    else:
        z_min = 0
        z_max = 1

    # Now, create frames with consistent z-axis limits
    for layer in range(num_layers):
        # Flatten the GM layer data
        GM_layer_flat = self.thermo_model.GM[layer].flatten()

        # Create a combined mask for valid coordinates and GM values
        GM_valid_mask = ~np.isnan(GM_layer_flat)
        combined_mask = valid_coords_mask & GM_valid_mask

        # Apply the combined mask
        x_layer = x_coords[combined_mask]
        y_layer = y_coords[combined_mask]
        z_layer = GM_layer_flat[combined_mask]

        # Check if there are enough points for triangulation
        if len(x_layer) >= 3:
            # Recompute Delaunay triangulation for this layer
            tri_layer = Delaunay(np.column_stack((x_layer, y_layer)))
            simplices_layer = tri_layer.simplices

            # Create the Mesh3d object for this frame
            mesh = go.Mesh3d(
                x=x_layer,
                y=y_layer,
                z=z_layer,
                i=simplices_layer[:, 0],
                j=simplices_layer[:, 1],
                k=simplices_layer[:, 2],
                intensity=z_layer,
                colorscale=colormap,
                showscale=True,  # Show the colorbar
                colorbar=dict(
                    title='GM [kJ/mol]',
                    tickfont=dict(size=12),
                ),
                cmin=z_min,  # Set the minimum of the color scale
                cmax=z_max,  # Set the maximum of the color scale
                flatshading=True,
                name=f'GM Layer {layer+1}'
            )

            # Append the frame
            frames.append(go.Frame(data=[mesh], name=f'Frame {layer+1}'))
        else:
            print(f"Layer {layer+1} has insufficient points for triangulation and will be skipped.")

    # Check if frames are generated
    if not frames:
        print("No valid frames were generated. Please check your data.")
    else:
        # Create the initial figure with the first valid frame
        fig = go.Figure(
            data=frames[0].data,  # Start with the first frame
            layout=go.Layout(
                title='GM over Ternary Diagram (Animated by Layer)',
                width=1000,  # Adjust as needed
                height=800,  # Adjust as needed
                scene=dict(
                    xaxis_title='Ternary X',
                    yaxis_title='Ternary Y',
                    zaxis_title='Mixing Free Energy [kJ/mol]',
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=2),
                    zaxis=dict(range=[z_min, z_max]),  # Set consistent z-axis limits
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [
                                [None], {
                                    'frame': {'duration': 0},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }
                            ]
                        }
                    ]
                }]
            ),
            frames=frames
        )

        # Add sliders for frame control
        fig.update_layout(
            sliders=[{
                'steps': [
                    {
                        'method': 'animate',
                        'args': [
                            [f'Frame {k+1}'],
                            {'mode': 'immediate', 'frame': {'duration': 0}, 'transition': {'duration': 0}}
                        ],
                        'label': f'{self.thermo_model.T_values[k]} K'
                    } for k in range(len(frames))
                ],
                'transition': {'duration': 0},
                'x': 0,
                'y': 0,
                'currentvalue': {'font': {'size': 12}, 'prefix': 'Layer: ', 'visible': True, 'xanchor': 'right'},
                'len': 1.0
            }],
        
        )

        # Show the figure
        if show_fig:
          fig.show()


  def ternary_plotly_GM_flat(self, colormap: str = 'rainbow', num_contours: int = 40, show_fig: bool = False):
    ''' create flat ternary diagrams of mixing free energy '''

    num_layers = len(self.thermo_model.T_values)

    # Flatten the ternary coordinates
    A = self.thermo_model.z[:,0]
    B = self.thermo_model.z[:,1]
    C = self.thermo_model.z[:,2]

    # Prepare a mask for valid ternary coordinates
    tolerance = 1e-6
    sum_ABC = A + B + C
    valid_coords_mask = (
        (A >= 0) & (B >= 0) & (C >= 0) &
        np.isclose(sum_ABC, 1.0, atol=tolerance)
    )

    # Convert ternary coordinates to 2D Cartesian coordinates
    def ternary_to_cartesian(a, b, c):
        x = 0.5 * (2 * b + c)  # Simplified since a + b + c = 1
        y = (np.sqrt(3) / 2) * c
        return x, y

    x_coords, y_coords = ternary_to_cartesian(A, B, C)

    # Initialize lists to collect GM values across layers
    all_z_values = []

    # Initialize frames
    frames = []

    for layer in range(num_layers):
        # Flatten and mask GM data
        GM_layer_flat = self.thermo_model.GM[layer].flatten()  # Shape: (121,)

        # Create per-layer valid mask
        per_layer_mask = valid_coords_mask & ~np.isnan(GM_layer_flat)  # Shape: (121,)

        # Apply the mask
        x_valid = x_coords[per_layer_mask]
        y_valid = y_coords[per_layer_mask]
        GM_valid = GM_layer_flat[per_layer_mask]

        # Interpolate GM values onto the grid
        grid_size = 100  # Adjust for resolution
        xi = np.linspace(0, 1, grid_size)
        yi = np.linspace(0, np.sqrt(3)/2, grid_size)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Mask to keep points inside the triangle
        triangle_mask = yi_grid <= (np.sqrt(3) * xi_grid)

        # Interpolate GM values onto the grid
        zi = griddata(
            (x_valid, y_valid), GM_valid, (xi_grid, yi_grid), method='linear'
        )

        # Mask out points outside the triangle
        zi[~triangle_mask] = np.nan

        # Collect z-values for global min and max
        all_z_values.extend(zi[~np.isnan(zi)])

    # Calculate global z-axis limits and color scale limits
    if all_z_values:
        all_z_values = np.array(all_z_values).flatten()
        # remove nan values
        all_z_values = all_z_values[~np.isnan(all_z_values)]
        # remove inf values
        all_z_values = all_z_values[~np.isinf(all_z_values)]

        z_min = np.min(all_z_values)
        z_max = np.max(all_z_values)
    else:
        z_min = 0
        z_max = 1

    # Now create frames for animation
    frames = []

    for layer in range(num_layers):
        # Flatten and mask GM data
        GM_layer_flat = self.thermo_model.GM[layer].flatten()

        # Create per-layer valid mask
        per_layer_mask = valid_coords_mask & ~np.isnan(GM_layer_flat)

        # Apply the mask
        x_valid = x_coords[per_layer_mask]
        y_valid = y_coords[per_layer_mask]
        GM_valid = GM_layer_flat[per_layer_mask]

        # Interpolate GM values onto the grid
        zi = griddata(
            (x_valid, y_valid), GM_valid, (xi_grid, yi_grid), method='linear'
        )

        # Mask out points outside the triangle
        zi[~triangle_mask] = np.nan

        # Create contour plot for this frame
        contour = go.Contour(
            x=xi,
            y=yi,
            z=zi,
            colorscale=colormap,
            zmin=z_min,
            zmax=z_max,
            ncontours=num_contours,
            contours=dict(showlines=False),
            colorbar=dict(title='GM [kJ/mol]'),
        )

        frame = go.Frame(data=[contour], name=f'Frame {layer+1}')
        frames.append(frame)

    # Create the initial figure with the first frame
    fig = go.Figure(data=frames[0].data)

    # Add frames to the figure
    fig.frames = frames

    # Update figure layout
    fig.update_layout(
        title='GM over Ternary Diagram (Animated by Layer)',
        width=800,
        height=700,
        xaxis=dict(
            title='',
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 1]
        ),
        yaxis=dict(
            title='',
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor='x',
            scaleratio=np.sqrt(3)/2,
            range=[0, np.sqrt(3)/2]
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [
                        None,
                        {'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True}
                    ]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [
                        [None],
                        {'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate', 'transition': {'duration': 0}}
                    ]
                }
            ]
        }],
        sliders=[{
            'steps': [
                {
                    'method': 'animate',
                    'args': [
                        [f'Frame {k+1}'],
                        {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True},
                        'transition': {'duration': 0}}
                    ],
                    'label': f'{self.thermo_model.T_values[k]} K'
                } for k in range(num_layers)
            ],
            'transition': {'duration': 0},
            'x': 0,
            'y': -0.1,
            'currentvalue': {'font': {'size': 12}, 'prefix': 'Layer: ', 'visible': True, 'xanchor': 'right'},
            'len': 1.0
        }]
    )

    # Show the figure
    if show_fig:
      fig.show()
