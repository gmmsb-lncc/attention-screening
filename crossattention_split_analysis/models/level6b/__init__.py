"""Level 6b: BAN + GRL (no cross-attention).

This is Level 5b (AttnPool + GRL) where the attention pooling +
concatenation + classifier is replaced by BAN fusion.

Hierarchy:
  - Level 5b: Proj → AttnPool → cat → classifier + GRL
  - **Level 6b**: Proj → **BAN** → classifier + GRL
"""

from .model import Level6bModel

__all__ = ["Level6bModel"]
