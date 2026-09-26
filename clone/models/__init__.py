"""Small black-box neural baselines."""

from .causal_lstm import CausalLSTM48
from .structured_residual import StructuredPlusResidual2k

__all__ = ["CausalLSTM48", "StructuredPlusResidual2k"]
