# CMA-DTI canonical integration

The official [qinchi1/CMA-DTI](https://github.com/qinchi1/CMA-DTI) repository is
pinned as the `CMA-DTI` submodule. The canonical wrapper preserves its GCN,
ChemBERTa/graph cross-attention, ESM-2 drug/protein attention and MLP decoder,
while enforcing this project's universal scaffold protocol.

Methodological corrections and provenance:

- checkpoint selection uses validation AUROC, matching upstream;
- the operating threshold maximizes MCC on validation and is frozen for test;
- test labels never participate in model or threshold selection;
- frozen encoders are evaluated deterministically and cached as ragged token
  features instead of being recomputed for every interaction and epoch;
- ESM-2 is `facebook/esm2_t33_650M_UR50D` because the architecture requires
  1280 channels; both encoder revisions are pinned in the YAML;
- `MAX_DRUG_NODES=310` retains all canonical molecules (the upstream 290 would
  reject four unique molecules in `human`/`all`);
- the virtual-node bit is used as the padding mask, correcting the upstream
  padded-node-count mask.

Install on an NVIDIA/CUDA 12.1 machine:

```bash
git submodule update --init CMA-DTI
bash environments/install_cmadti_cuda.sh
```

If a previous installation stopped midway, rebuild the partial environment:

```bash
bash environments/install_cmadti_cuda.sh --force
```

If all packages were installed but the final DGL/GraphBolt verification failed,
repair the compatibility replacement and rerun the smoke test without
reinstalling the environment:

```bash
bash environments/install_cmadti_cuda.sh --repair
```

The installer initializes the pinned submodule automatically and skips its
unused Git-LFS example datasets. `networkx` is deliberately installed by Conda
before PyTorch/DGL to prevent pip/conda path collisions.

Each installation uses a fresh temporary Conda package cache.  This prevents a
partial or corrupted package in the host's global cache from contaminating the
CMA-DTI environment.  The GraphBolt compatibility replacement is also applied
without modifying Conda's hard-linked package cache, and its stale precompiled
bytecode is removed before verification.

The installer ends with a synthetic CUDA forward/backward through DGL, GCN,
both attention blocks and the decoder before any long training run starts.

Run all five seeds for one or more corpora:

```bash
CMADTI_CORPORA="human non_human" \
CMADTI_SEEDS="42 123 456 789 1024" \
bash scripts/cmadti/run_canonical_cuda.sh
```

Use `CMADTI_CORPORA=all` on the second machine. Feature caches are stored under
`results/cmadti/feature_cache/<corpus>` and reused across seeds. A completed
checkpoint is also reused, allowing the runner to backfill evaluation files.

Each seed emits `best_model.pt`, `raw_predictions.npz`,
`cmadti_calibration.json`, and `cmadti_results.json`. After the requested seeds,
`cmadti_<corpus>_aggregate.json` contains mean and population standard
deviation.

After training, CMA-DTI can be selected independently with
`--models dtkinase,drugban,graphban,conplex,cmadti`, or together with ChemGLaM
through the experimental `--profile full_6model`. The four-model canonical
profile remains unchanged until the expanded panel is statistically validated.
