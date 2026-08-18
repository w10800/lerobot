"""Target-time conditioning used by SnapFlow.

The paper specifies a zero-initialized two-layer MLP. We zero-initialize the
second layer so the module is exactly output-preserving at construction while
still allowing the output layer to receive a gradient on the first update.
"""

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class ZeroInitTargetTimeMLP(nn.Module):
    """Project an encoded target time into the existing time-embedding space."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, encoded_target_time: Tensor) -> Tensor:
        return self.out_proj(F.silu(self.in_proj(encoded_target_time)))
