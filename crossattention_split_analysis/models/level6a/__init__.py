"""Level 6a: CrossAttn + BAN + GRL.

This is Level 5a (CrossAttn + AttnPool + GRL) where the concatenation
+ classifier is replaced by Bilinear Attention Network (BAN) fusion.

Hierarchy:
  - Level 5a: Proj → CrossAttn → AttnPool → cat → classifier + GRL
  - **Level 6a**: Proj → CrossAttn → **BAN** → classifier + GRL
"""

from .model import Level6aModel

__all__ = ["Level6aModel"]
