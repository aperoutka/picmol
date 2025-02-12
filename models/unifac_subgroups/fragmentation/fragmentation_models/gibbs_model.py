"""gibbs model"""

from typing import List, Union
import pandas as pd
from .fragmentation_model import *


class GibbsModel(FragmentationModel):

	"""GibbsModel it's a fragmentation model dedicated to Gibbs excess models."""
	"""unifac is an instance of this class."""


	def __init__(
		self,
		subgroups: pd.DataFrame,
		RQ: pd.DataFrame,
		split_detection_smarts: List[str] = [],
		problematic_structures: Union[pd.DataFrame, None] = None,
	) -> None:
		super().__init__(subgroups, RQ, split_detection_smarts, problematic_structures)