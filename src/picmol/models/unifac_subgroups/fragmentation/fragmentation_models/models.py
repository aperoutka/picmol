import pandas as pd
from pathlib import Path
from .gibbs_model import *


_here = Path(__file__).parent.parent.parent # path of python script
_problems = pd.read_csv(_here / "problematic_structures.tsv", sep="\t", index_col="smarts", comment="?")

# =============================================================================
# UNIFAC
# =============================================================================

_uni = pd.read_csv(_here / "unifac_subgroup_info.csv", index_col="subgroup")
_uni_rq = pd.read_csv(_here / "unifac_subgroups.csv", index_col="subgroup")

unifac = GibbsModel(
  subgroups=_uni,
  RQ=_uni_rq,
  split_detection_smarts=["C5H4N", "C5H3N", "C4H3S", "C4H2S"],
  problematic_structures=_problems,
)

_uni_IL = pd.read_csv(_here / "unifac_il_subgroup_info.csv", index_col="subgroup")
_uni_IL_rq = pd.read_csv(_here / "unifac_il_subgroups.csv", index_col="subgroup")

unifac_IL = GibbsModel(
  subgroups=_uni_IL,
  RQ=_uni_IL_rq,
  split_detection_smarts=["C5H4N", "C5H3N", "C4H3S", "C4H2S"],
  problematic_structures=_problems,
)