# Diffusion vs CNN Cross-Attention: Why Similar Datasets Favor Diffusion

This note explains, in technical terms, why diffusion-based classifiers can outperform CNN+Cross-Attention when the dataset contains **highly similar samples** and class boundaries are defined by **subtle differences**. It is intended to complement the diffusion guide and clarify the practical advantage observed in our experiments.

## Executive Summary

- **CNN encoders** prioritize **frequent local patterns**, which tend to represent the **common structure** across samples (the "general morphology").
- In **high-similarity datasets**, the class signal is often encoded in **small, distributed deviations** (micro-differences).
- **Diffusion training** forces the model to **reconstruct the original signal under noise**, preserving micro-differences that CNNs often smooth out.
- The result is **better separability** and higher MCC in scenarios where positive and negative examples are very close in representation space.

---

## 1) What "General Morphology" Means (Formally)

Let $x \in \mathbb{R}^{T \times d}$ be a token-level embedding matrix (protein or ligand).  
CNN encoders apply local convolutions and pooling-like operations that are **biased toward patterns with high frequency** across the dataset.

A local convolution can be written as:

$$
z_t = \sigma\left(\sum_{k=-K}^{K} W_k x_{t+k}\right)
$$

When many samples are similar, the convolution learns filters that respond strongly to **shared motifs**.  
This yields a representation dominated by **average local structure**, i.e. the *morphology common to all samples*.

If the label is determined by a **small perturbation** $\delta$:

$$
x^+ = x + \delta
$$

then local filters may suppress $\delta$ if it does not consistently align with learned kernels:

$$
z_t(x + \delta) \approx z_t(x)
$$

This makes **class-conditional separation** weak.

---

## 2) Why Diffusion Preserves Micro-Differences

In diffusion training, the model must **recover the clean signal** under noise:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

The denoiser predicts $\epsilon_\theta(x_t, t)$, and the training loss penalizes reconstruction error:

$$
L_{diff} = \mathbb{E}_{t,\epsilon} \left[ w(t)\|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]
$$

Because the model must reconstruct $x_0$ from $x_t$, **any micro-difference** $\delta$ becomes part of the signal that must remain recoverable:

$$
x_0 + \delta \Rightarrow x_t + \sqrt{\bar{\alpha}_t}\,\delta
$$

Thus the training objective implicitly forces the model to **retain small discriminative components**, instead of smoothing them out.

---

## 3) Separability Argument (Why MCC Improves)

Assume embeddings of two classes are close:

$$
z^+ \sim \mathcal{N}(\mu + \delta, \Sigma), \quad
z^- \sim \mathcal{N}(\mu, \Sigma)
$$

CNN encoders tend to reduce $\|\delta\|$ by smoothing local patterns.  
Diffusion-based denoising preserves $\delta$ since it must remain reconstructable.  
Therefore the class means remain more separated in diffusion:

$$
\|\mu^+ - \mu^-\|_{\text{diffusion}} > \|\mu^+ - \mu^-\|_{\text{cnn}}
$$

Higher separability directly improves MCC and threshold stability.

---

## 4) Cross-Attention After Denoising

In our pipeline, cross-attention is applied **after** denoising, so interaction modeling uses **cleaner representations**:

$$
H_p' = \text{CrossAttn}(H_p, H_l), \quad H_l' = \text{CrossAttn}(H_l, H_p)
$$

This is beneficial in high-similarity regimes because the attention mechanism is not dominated by noise or oversmoothed features.

---

## Practical Implication

When the dataset has many near-duplicate or highly similar samples, **CNNs capture the common structure but often miss the marginal signal** that defines the label.  
Diffusion training makes that marginal signal recoverable, improving class separability and MCC.

---

## Where to Learn More

- Diffusion pipeline details: `crossattention_split_analysis/DIFFUSION.md`
