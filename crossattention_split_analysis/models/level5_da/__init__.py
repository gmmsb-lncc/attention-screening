"""Level 5-DA: Cross-Attention + Domain Adaptation (GRL).

Extends Level 5-Lite (Level 4 in the benchmark) with adversarial
domain adaptation via a Gradient Reversal Layer (GRL).  The domain
discriminator is trained to predict scaffold cluster membership,
while the GRL forces the feature extractor to produce
scaffold-invariant representations.

Reference: Ganin et al., "Domain-Adversarial Training of Neural
Networks", JMLR 2016.

Ablation hierarchy:
  Level 4 = Cross-Attention encoder + KNN/MLP
  Level 5 = Level 4 + Gradient Reversal (scaffold-invariant features)
"""

from .model import Level5DAModel
from .domain_adaptation import (
    GradientReversalLayer,
    DomainDiscriminator,
    build_scaffold_clusters,
    lambda_schedule,
)

__all__ = [
    "Level5DAModel",
    "GradientReversalLayer",
    "DomainDiscriminator",
    "build_scaffold_clusters",
    "lambda_schedule",
]
