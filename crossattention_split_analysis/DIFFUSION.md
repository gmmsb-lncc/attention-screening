# Diffusion Variant: Detailed Guide

This document explains the `diffusion` model variant added to the CrossAttention Split Analysis pipeline. The goal is to keep **token-level protein and ligand matrices** intact, denoise them with a lightweight diffusion encoder, and then perform classification/regression with minimal information loss via attention pooling.

## When To Use

Use `--model_variant diffusion` if you want a **matrix-preserving** classifier that:
- Operates directly on per-token embeddings (protein and ligand).
- Learns a denoising objective to regularize representations.
- Avoids aggressive pooling early in the network.

This is especially useful when you suspect important signal is distributed across many tokens and could be lost by mean pooling.

## High-Level Idea

We treat each embedding matrix as a sequence of tokens and apply a **denoising diffusion model (DDPM-style)** to it. During training, we add noise at a random timestep and train a denoiser to predict the noise. The denoised matrix is then used for classification.

### Data Flow (Visual)

```
Protein matrix P (Lp x Dp)     Ligand matrix L (Ll x Dl)
          |                               |
          v                               v
   Linear projection                Linear projection
          |                               |
          v                               v
   Diffusion denoiser               Diffusion denoiser
   (Transformer encoder)            (Transformer encoder)
          |                               |
          v                               v
   Attention pooling                Attention pooling
          |                               |
          +----------- concat ------------+
                          |
                          v
                  Multi-task head
              (classification + regression)
```

## Mathematical Formulation

### Forward Diffusion (Noise Injection)

For a clean matrix \(x_0\) (protein or ligand), we sample a timestep \(t\) and apply:

\[
q(x_t | x_0) = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon
\]

- \(\epsilon \sim \mathcal{N}(0, I)\)
- \(\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)\)

### Denoising Objective

The denoiser \(\epsilon_\theta(x_t, t)\) predicts the injected noise:

\[
\mathcal{L}_{diff} = \mathbb{E}_{t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2\right]
\]

We apply this loss **separately** to protein and ligand matrices, then sum them.

### Reconstruction (Used for Classifier Input)

We recover an estimate of \(x_0\):

\[
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}
\]

### Final Objective

The final training objective combines classification/regression loss and diffusion loss:

\[
\mathcal{L} = \alpha\,\mathcal{L}_{cls} + \beta\,\mathcal{L}_{reg} + \lambda\,\mathcal{L}_{diff}
\]

Where \(\lambda\) is controlled by `--diffusion_loss_weight`.

## Pooling Strategy (Why Information Is Preserved)

Instead of mean pooling, we use **attention pooling**:

\[
\text{score}_i = q^\top W x_i
\]
\[
\text{attn}_i = \text{softmax}(\text{score}_i)
\]
\[
\text{pool}(x) = \sum_i \text{attn}_i x_i
\]

This allows the model to learn **which tokens matter most** without discarding token-level information early.

## Implementation Details

- **Denoiser**: `TransformerEncoder` with time embeddings.
- **Pooling**: learned attention pooling per modality.
- **Auxiliary loss**: diffusion MSE loss, added only during training.
- **No cross-attention** between protein and ligand in this variant.
- **Compatible with existing split analysis pipeline.**

Key file:
- `src/classifier/models/diffusion_model.py`

## CLI Usage

### Example: Non-human dataset with 8M ESM-2 and MoLFormer ligand matrices

```bash
python crossattention_split_analysis_main.py \
  --embedding 8M \
  --dataset non_human \
  --model_variant diffusion \
  --molformer_ligand \
  --scaffold_split_dir scaffolds_splits/output \
  --output_dir results/crossattention_analysis_diffusion \
  --epochs 500 \
  --patience 30 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --hidden_dim 256 \
  --num_heads 8 \
  --ff_dim 1024 \
  --dropout 0.1 \
  --diffusion_steps 200 \
  --diffusion_layers 4 \
  --diffusion_loss_weight 0.1 \
  --classification_weight 1.0 \
  --regression_weight 0.5 \
  --threshold_metric mcc \
  --seeds 42 123 456 789 1024 \
  --force
```

## Practical Tips

- If training is unstable, try lowering `--diffusion_loss_weight`.
- Increasing `--diffusion_steps` improves denoising but adds computation.
- Use `--molformer_ligand` if only MoLFormer matrices are available.

## Limitations (Current)

- No explicit protein-ligand cross-attention in this variant.
- Diffusion is applied independently to protein and ligand matrices.
- Attention pooling can still compress information, but it is learned and token-aware.

## Next Possible Extension

To move toward the hybrid approach (diffusion + cross-attention), you can:
- Denoise protein/ligand matrices with diffusion.
- Feed denoised matrices into the existing cross-attention blocks.

This would preserve token-level structure while allowing **interaction modeling** across modalities.
