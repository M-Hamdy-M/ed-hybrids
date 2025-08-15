# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

from easydict import EasyDict
from .mlp import OptimizableMLP
from .momlp import OptimizableMOMLP
from .ed import OptimizableEdRVFL, OptimizableSS


optimizable_models = EasyDict(
      EdRVFL=OptimizableEdRVFL,
      MLP=OptimizableMLP,
      SS=OptimizableSS,
      MOMLP=OptimizableMOMLP,
)