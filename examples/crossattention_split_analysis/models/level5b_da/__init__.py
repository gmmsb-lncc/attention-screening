"""Level 5b-DA: Attention Pooling + Adversarial Domain Adaptation.

This is Level 3 (AttnPool) augmented with a Gradient Reversal Layer
for scaffold-invariant representations.  Unlike Level 5 (which is
Level 4 / Cross-Attention + GRL), Level 5b isolates the effect of
domain adaptation on an architecture *without* cross-attention.

Hierarchy:
  - Level 3:  Projection + AttnPool → KNN/MLP
  - Level 5b: Projection + AttnPool + **GRL** → KNN/MLP   ← this module
  - Level 4:  Projection + CrossAttn + AttnPool → KNN/MLP
  - Level 5:  Projection + CrossAttn + AttnPool + **GRL** → KNN/MLP
"""

from .model import Level5bDAModel

__all__ = ["Level5bDAModel"]
