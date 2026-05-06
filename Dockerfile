# =============================================================================
# attention-screening — 4-model committee (DT-Kinase, DrugBAN, GraphBAN, ConPLex)
# =============================================================================
#
# Single Conda env "baseline" hosting all 4 models. Inference-only image.
#
# Build:
#   docker build -t attention-screening:cpu --build-arg BUILD_TYPE=cpu .
#   docker build -t attention-screening:cuda --build-arg BUILD_TYPE=cuda .
#
# Run (lookup demo, no GPU):
#   docker run --rm attention-screening:cpu \
#       bash scripts/inference/examples/run_imatinib_demo.sh
#
# Run (custom SMILES, with HF cache volume):
#   docker run --rm -v hf-cache:/root/.cache/huggingface \
#       -v $PWD/results:/app/results attention-screening:cpu \
#       "CC(=O)Oc1ccccc1C(=O)O" --organism human --ckpt-corpus all
#
# Run (CUDA, GPU):
#   docker run --rm --gpus all -v hf-cache:/root/.cache/huggingface \
#       attention-screening:cuda "<SMILES>"
# =============================================================================

ARG BUILD_TYPE=cpu

FROM continuumio/miniconda3:24.9.2-0

ARG BUILD_TYPE
ENV BUILD_TYPE=${BUILD_TYPE}

WORKDIR /app

# OS deps for RDKit + git (some HF model downloads use git-lfs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage A: install conda env (heaviest layer; cached via setup script + reqs)
# -----------------------------------------------------------------------------
COPY scripts/inference/setup_baseline_env.sh scripts/inference/setup_baseline_env.sh

RUN if [ "${BUILD_TYPE}" = "cuda" ]; then \
        bash scripts/inference/setup_baseline_env.sh ; \
    else \
        bash scripts/inference/setup_baseline_env.sh --cpu ; \
    fi \
    && conda clean -afy \
    && find /opt/conda -name "*.pyc" -delete \
    && find /opt/conda -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage B: copy repo (ckpts + scripts + scaffold splits)
# -----------------------------------------------------------------------------
COPY . /app

# -----------------------------------------------------------------------------
# Stage C: clone upstream baselines (DrugBAN + GraphBAN). ConPLex/src is
# vendored in-repo, so no clone needed.
# -----------------------------------------------------------------------------
RUN rm -rf /app/DrugBAN/src \
    && git clone --depth=1 https://github.com/peizhenbai/DrugBAN.git /app/DrugBAN/src \
    && rm -rf /app/GraphBAN/src \
    && git clone --depth=1 https://github.com/HamidHadipour/GraphBAN.git /app/GraphBAN/src \
    && conda run --no-capture-output -n baseline \
        python /app/GraphBAN/patch_upstream.py --src /app/GraphBAN/src \
    && rm -rf /app/DrugBAN/src/.git /app/GraphBAN/src/.git

# Make conda env default for CMD/ENTRYPOINT
ENV PATH=/opt/conda/envs/baseline/bin:${PATH}
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# HuggingFace cache lives in /root/.cache/huggingface — mount as volume.
VOLUME ["/root/.cache/huggingface", "/app/results"]

# Default entrypoint: attention_screening.py.
# Pass SMILES + flags as docker run args.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "baseline", \
            "python", "attention_screening.py"]
CMD ["--help"]
