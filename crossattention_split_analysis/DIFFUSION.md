# Diffusion Variant: Detailed Guide

This document explains the `diffusion` model variant added to the CrossAttention Split Analysis pipeline. The goal is to keep **token-level protein and ligand matrices** intact, denoise them with a lightweight diffusion encoder, and then perform classification/regression with minimal information loss via attention pooling.

## When To Use

Use `--model_variant diffusion` if you want a **matrix-preserving** classifier that:
- Operates directly on per-token embeddings (protein and ligand).
- Learns a denoising objective to regularize representations.
- Avoids aggressive pooling early in the network.

This is especially useful when you suspect important signal is distributed across many tokens and could be lost by mean pooling.

## High-Level Idea

We treat each embedding matrix as a sequence of tokens and apply a **denoising diffusion model (DDPM-style)** to it. During training, we add noise at a random timestep and train a denoiser to predict the noise. The denoised matrix is then used for classification. The current implementation adds **lightweight cross-attention** after denoising to explicitly model protein–ligand interaction.

## Didactic Flow (What Happens First, Step by Step)

The protein and ligand matrices are **not concatenated at the input**. They are processed **separately** through diffusion, then **interacted** via cross-attention, and **only at the very end** their pooled vectors are concatenated for classification.

```
Protein matrix P (Lp x Dp)                     Ligand matrix L (Ll x Dl)
         |                                            |
  Linear + LayerNorm                           Linear + LayerNorm
         |                                            |
 + Positional Encoding (scaled)              + Positional Encoding (scaled)
         |                                            |
   Add noise (diffusion step)                Add noise (diffusion step)
         |                                            |
  Denoiser (Transformer)                    Denoiser (Transformer)
         |                                            |
  Reconstruct x0_hat                         Reconstruct x0_hat
         |                                            |
         +----------- Cross-Attention --------------+
                     (protein ↔ ligand)
         |                                            |
  Multi‑Query Pooling                         Multi‑Query Pooling
         |                                            |
          \_________________ concat __________________/
                            |
                    Classification head
```

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
   Cross-attention block           Cross-attention block
   (protein ↔ ligand)              (protein ↔ ligand)
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

To improve classification stability and avoid over-emphasizing highly noisy timesteps, the loss is **SNR‑weighted**:

\[
w(t) = \log(1 + \text{SNR}_t)
\]
\[
\mathcal{L}_{diff} = \mathbb{E}_{t,\epsilon}\left[w(t)\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2\right]
\]

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
Optionally, \(\lambda\) can be **annealed** linearly during training (`--diffusion_loss_anneal linear`), starting high and decaying to zero.

## Pooling Strategy (Why Information Is Preserved)

Instead of mean pooling, we use **multi‑query attention pooling** to capture multiple salient regions:

\[
\text{score}_{i,k} = q_k^\top W x_i
\]
\[
\text{attn}_{i,k} = \text{softmax}(\text{score}_{i,k})
\]
\[
\text{pool}(x) = \frac{1}{K}\sum_k \sum_i \text{attn}_{i,k} x_i
\]

This allows the model to learn **multiple token‑level focuses** without discarding information early.

## Implementation Details

- **Denoiser**: separate `TransformerEncoder` for protein and ligand, with time embeddings.
- **Positional encoding**: **sinusoidal** PE added after projection, with learnable scale.
- **Normalization**: `LayerNorm` applied after projection per modality.
- **Pooling**: multi‑query attention pooling per modality (`--diffusion_pool_queries`).
- **Auxiliary loss**: SNR‑weighted diffusion MSE loss, added only during training. The sampling bias and weight strength can be tuned with `--diffusion_snr_sampling_gamma` and `--diffusion_snr_sampling_mix`.
- **Cross‑attention**: lightweight protein↔ligand block(s) after denoising (`--diffusion_cross_attn_layers`).
- **Classification-only mode**: optional regressor removal for pure classification (`--classification_only`).
- **Compatible with existing split analysis pipeline.**

Key file:
- `src/classifier/models/diffusion_model.py`

## Why ~3.9M Parameters? (Didactic Breakdown)

The diffusion model includes a **Transformer denoiser**, which dominates the parameter count. A rough breakdown for the default config is:

- **Hidden dim**: `d = 256`
- **FF dim**: `ff = 1024`
- **Layers**: `L = 4`

### 1) Transformer Encoder (per layer)

Each layer has:

**Multi-Head Attention**

- Q/K/V projections: `3 * d * d`
- Output projection: `1 * d * d`

Total: `4 * d^2 = 4 * 256^2 = 262,144`

**Feed-Forward (MLP)**

- First linear: `d * ff = 256 * 1024 = 262,144`
- Second linear: `ff * d = 1024 * 256 = 262,144`

Total: `2 * d * ff = 524,288`

**LayerNorm + biases** are small compared to the above.

**Per-layer total (approx):**  
`262,144 + 524,288 = 786,432` params

**All layers (L=4):**  
`~ 3.15M` params

### 2) Projections (protein + ligand)

- Protein projection: `320 -> 256` → `320 * 256 + 256 ≈ 82k`
- Ligand projection: `768 -> 256` → `768 * 256 + 256 ≈ 197k`

**Total:** `~279k`

### 3) Attention Pooling + Multi-Task Head

- Attention pool (per modality): ~`256 * 256` each
- Multi-task head (two-layer MLPs): ~`200k+`

### Total (expected)

Summing these pieces gives **~3.5M–4.0M**, which matches the observed log value:

```
Model parameters: 3,885,570
```

So the parameter count is expected and mainly comes from the diffusion denoiser.

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
  --diffusion_cross_attn_layers 1 \
  --diffusion_pool_queries 4 \
  --diffusion_snr_sampling_gamma 0.5 \
  --diffusion_snr_sampling_mix 0.2 \
  --diffusion_loss_weight 0.05 \
  --diffusion_loss_anneal linear \
  --classification_only \
  --classification_weight 1.0 \
  --regression_weight 0.0 \
  --threshold_metric mcc \
  --seeds 42 123 456 789 1024 \
  --force
```

## Practical Tips

- If training is unstable, try lowering `--diffusion_loss_weight` or enabling `--diffusion_loss_anneal linear`.
- If MCC stalls, increase `--diffusion_pool_queries` (e.g., 8) or `--diffusion_cross_attn_layers`.
- Increasing `--diffusion_steps` improves denoising but adds computation.
- Use `--molformer_ligand` if only MoLFormer matrices are available.

## Limitations (Current)

- Cross‑attention is lightweight (not full cross‑attention stack).
- Diffusion is applied independently per modality before interaction.
- Attention pooling can still compress information, but it is learned and token-aware.

## Next Possible Extension

To move toward the hybrid approach (diffusion + cross-attention), you can:
- Denoise protein/ligand matrices with diffusion.
- Feed denoised matrices into the existing cross-attention blocks.

This would preserve token-level structure while allowing **interaction modeling** across modalities.
