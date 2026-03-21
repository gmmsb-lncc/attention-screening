"""Test script to verify Level 3 model is being used correctly."""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'crossattention_split_analysis')

from models.level3_crossatt import Level3CrossAttModel
from crossattention_split_analysis.config import TrainingConfig

# Simulate what run_scenario does
config = TrainingConfig(
    protein_dim=320,
    ligand_dim=768,
    hidden_dim=256,  # CRITICAL: This should be 256
    num_heads=8,
    dropout=0.4,
    classifier_dropout=0.6,
    model_variant="level3_crossatt",
)

print(f"Config hidden_dim: {config.hidden_dim}")
print(f"Config model_variant: {config.model_variant}")

# Create model
model = Level3CrossAttModel(
    protein_dim=config.protein_dim,
    ligand_dim=config.ligand_dim,
    hidden_dim=config.hidden_dim,
    num_heads=config.num_heads,
    encoder_dropout=config.dropout,
    attention_dropout=config.dropout,
    classifier_dropout=config.classifier_dropout,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")
print(f"Expected: ~972,545")
print(f"Match: {abs(n_params - 972_545) < 1000}")
