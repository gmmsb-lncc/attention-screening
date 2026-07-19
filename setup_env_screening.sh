#!/usr/bin/env bash
# =============================================================================
# setup_env_screening.sh
#
# Build ONE unified conda environment named "screening" that can run the four
# protein-ligand interaction models developed in this repository:
#
#   DT-Kinase (v7)  - ESM-2 8M + MoLFormer + CNN 2D cross-attention (torch only)
#   ConPLex         - ProtBert + Morgan FP contrastive co-embedding (torch only)
#   DrugBAN         - CNN-1D + DGL/dgllife GCN bilinear attention (needs DGL)
#   GraphBAN        - dgllife GCN + ESM-1b + ChemBERTa BAN         (needs DGL)
#
# Scope: INFERENCE (committee) - imports and runs the four scripts under
#        scripts/inference/models/{dtkinase,conplex,drugban,graphban}_score.py
#        and the committee aggregation. It is NOT a from-scratch training env.
#
# Portable / auto-detecting. Three install profiles:
#
#   macOS arm64 (Apple Silicon, e.g. M1 Pro)
#       torch 2.3.1 + dgl 2.3.0 from conda-forge (the only self-consistent
#       osx-arm64 pair; pip torch 2.4.1 has NO ABI-matched DGL on arm64).
#       torch models run on the M1 GPU via the MPS backend; DGL graph ops
#       run on CPU (DGL has no MPS/CUDA backend on mac regardless).
#       ALL FOUR models run here.
#
#   Linux + NVIDIA GPU (diamante hosts)
#       torch 2.4.1 (cu121) + DGL from dglteam/label/cu121 - the exact,
#       validated baseline pins. ALL FOUR models run, fidelity-faithful.
#
#   Linux CPU-only (fallback)
#       torch 2.4.1 (cpu) + DGL from conda-forge.
#
# Reconciled pins shared by every profile (derived from all four models):
#   python=3.10   transformers==4.39.3   numpy<2   pandas<2.1
#   huggingface_hub<1.0   setuptools<71   sentencepiece   protobuf<5   rdkit
#
#   - transformers==4.39.3: MoLFormer-XL remote code imports transformers.onnx
#     (removed in >=4.40); ConPLex needs <4.46; 4.39.3 is the unique
#     intersection that also loads ProtBert (slow tokenizer) and ChemBERTa.
#   - pandas<2.1: rdkit PandasTools (ConPLex import chain) breaks on 2.1+.
#     DT-Kinase does not require >=2.1 (that pin lives only in environment.yml;
#     the pip requirement files pin >=1.5.3).
#
# ESM handling: DT-Kinase and GraphBAN both `import esm`. The pip `fair-esm`
# wheel segfaults on macOS, so this script uses a LOCAL clone at llm/ESM and
# registers it at the FRONT of sys.path for every interpreter in the env via a
# .pth startup hook. No pip fair-esm is installed.
#
# Usage:
#   bash setup_env_screening.sh              # all four models (default)
#   SCREENING_MODELS="dtkinase conplex" bash setup_env_screening.sh
#                                            # torch-only pair; skips DGL. On
#                                            # mac this uses pip torch 2.4.1
#                                            # (strict fidelity, no DGL needed).
#   SCREENING_ENV_NAME=myenv bash setup_env_screening.sh
#
# Env knobs:
#   SCREENING_ENV_NAME   (default "screening")
#   SCREENING_MODELS     (default "dtkinase conplex drugban graphban")
#   SCREENING_SKIP_CLONES=1   skip cloning ESM / DrugBAN / GraphBAN upstream src
#   SCREENING_SKIP_VERIFY=1   skip the final import smoke test
# =============================================================================
set -euo pipefail

ENV_NAME="${SCREENING_ENV_NAME:-screening}"
MODELS="${SCREENING_MODELS:-dtkinase conplex drugban graphban}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '\033[1;36m[screening]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[screening WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[screening ERROR]\033[0m %s\n' "$*" >&2; }

has_model() { case " ${MODELS} " in *" $1 "*) return 0;; *) return 1;; esac; }

INCLUDE_DGL=false
if has_model drugban || has_model graphban; then INCLUDE_DGL=true; fi

# ---------------------------------------------------------------------------
# 0. Platform detection
# ---------------------------------------------------------------------------
OS="$(uname -s)"; ARCH="$(uname -m)"
GPU_MODE=false; CUDA_VER=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_MODE=true
  CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
fi

PROFILE=""
if [ "${OS}" = "Darwin" ] && [ "${ARCH}" = "arm64" ]; then
  PROFILE="mac_arm64"
elif [ "${OS}" = "Linux" ] && [ "${GPU_MODE}" = true ]; then
  PROFILE="linux_gpu"
else
  PROFILE="cpu"
fi

log "Repo         : ${REPO}"
log "Env name     : ${ENV_NAME}"
log "Models       : ${MODELS}  (DGL baselines: ${INCLUDE_DGL})"
log "Platform     : ${OS}/${ARCH}  GPU=${GPU_MODE}${CUDA_VER:+ (CUDA ${CUDA_VER})}  -> profile '${PROFILE}'"

# ---------------------------------------------------------------------------
# 1. conda hook
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  err "conda not found in PATH. Install Miniconda/Anaconda/Miniforge first."
  exit 1
fi
# conda's shell functions dereference unbound vars (PS1, etc.); relax nounset
# and errexit around the hook + activate or the script dies here silently.
set +u
eval "$(conda shell.bash hook)"

if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Environment '${ENV_NAME}' already exists - reusing (packages will be added/updated)."
else
  log "Creating environment '${ENV_NAME}' (python 3.10)..."
  conda create -y -n "${ENV_NAME}" -c conda-forge python=3.10
fi
conda activate "${ENV_NAME}"
set -u
[ "${CONDA_DEFAULT_ENV:-}" = "${ENV_NAME}" ] || { err "conda activate ${ENV_NAME} failed (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset})."; exit 1; }
PY="python3"
log "Active env: ${CONDA_DEFAULT_ENV}  ($(command -v ${PY}))"

site_packages() { ${PY} -c "import site; print(site.getsitepackages()[0])"; }

# ---------------------------------------------------------------------------
# 2. Core stack per profile (numpy<2 / pandas<2.1 are the ABI anchor)
# ---------------------------------------------------------------------------
CONDA_BASE_PKGS=(
  "numpy>=1.26,<2" "pandas>=1.5,<2.1" scipy "scikit-learn>=1.5"
  rdkit h5py pyyaml tqdm prettytable yacs "setuptools>=65,<71" networkx
)

install_torch_dgl() {
  case "${PROFILE}" in
    mac_arm64)
      if [ "${INCLUDE_DGL}" = true ]; then
        # Only self-consistent osx-arm64 set: conda-forge pytorch 2.3.1 (cpu+MPS)
        # + dgl 2.3.0 (graphbolt built against torch 2.3.1). Solve them together.
        log "conda-forge stack: pytorch 2.3.1 (cpu/MPS) + dgl 2.3.0 + scientific base (single solve)..."
        conda install -y -n "${ENV_NAME}" -c conda-forge \
          "pytorch=2.3.1=*cpu*" "dgl=2.3.0" "${CONDA_BASE_PKGS[@]}"
      else
        log "conda-forge scientific base + pip torch 2.4.1 (arm64 CPU/MPS wheel, strict pins, no DGL)..."
        conda install -y -n "${ENV_NAME}" -c conda-forge "${CONDA_BASE_PKGS[@]}"
        pip install -c "${CONSTRAINTS}" torch==2.4.1
      fi
      ;;
    linux_gpu)
      log "conda-forge scientific base (no torch, no dgl)..."
      conda install -y -n "${ENV_NAME}" -c conda-forge "${CONDA_BASE_PKGS[@]}"
      local pip_cuda="cu121"; local dgl_label="cu121"
      if [ -n "${CUDA_VER}" ] && [ "${CUDA_VER%%.*}" -lt 12 ]; then pip_cuda="cu118"; dgl_label="cu118"; fi
      log "pip torch 2.4.1 (${pip_cuda}) from pytorch.org..."
      pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
      pip install -c "${CONSTRAINTS}" torch==2.4.1 \
        --index-url "https://download.pytorch.org/whl/${pip_cuda}"
      if [ "${INCLUDE_DGL}" = true ]; then
        log "DGL from dglteam/label/${dgl_label} (--no-update-deps so pip torch is preserved)..."
        pip uninstall -y dgl 2>/dev/null || true
        conda install -y -n "${ENV_NAME}" dgl -c "dglteam/label/${dgl_label}" --no-update-deps
      fi
      ;;
    cpu)
      log "conda-forge scientific base + CPU torch/dgl..."
      if [ "${INCLUDE_DGL}" = true ]; then
        conda install -y -n "${ENV_NAME}" -c conda-forge \
          "pytorch=2.3.1=*cpu*" "dgl" "${CONDA_BASE_PKGS[@]}"
      else
        conda install -y -n "${ENV_NAME}" -c conda-forge "${CONDA_BASE_PKGS[@]}"
        pip install -c "${CONSTRAINTS}" torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
      fi
      ;;
  esac
}

# pip constraints: forbid any transitive dep from bumping the ABI anchors.
# torch is pinned to whatever is already installed so pip never re-resolves it.
CONSTRAINTS="$(mktemp -t screening-constraints.XXXXXX)"
write_constraints() {
  local torch_ver
  torch_ver="$(${PY} -c 'import torch;print(torch.__version__.split("+")[0])' 2>/dev/null || true)"
  {
    echo "numpy<2"
    echo "pandas<2.1"
    echo "huggingface_hub<1.0"
    echo "protobuf<5"
    echo "setuptools<71"
    if [ -n "${torch_ver}" ]; then echo "torch==${torch_ver}"; fi
  } > "${CONSTRAINTS}"
}
# seed constraints before the (possible) pip-torch step, then rewrite once torch exists
write_constraints
install_torch_dgl
write_constraints

log "Torch check: $(${PY} -c 'import torch;print(torch.__version__)')"

# ---------------------------------------------------------------------------
# 3. HuggingFace + ConPLex/GraphBAN pip stack (constrained; won't move anchors)
# ---------------------------------------------------------------------------
log "pip: transformers==4.39.3 + tokenizer/co-embedding deps (constrained)..."
pip install -c "${CONSTRAINTS}" \
  transformers==4.39.3 safetensors sentencepiece "protobuf<5" \
  "huggingface_hub<1.0" omegaconf

if has_model conplex; then
  # ConPLex src/featurizers/molecule.py + protein.py have BARE module-level
  # imports of pysmiles / deepchem / dscript even though the Morgan+ProtBert
  # inference path never calls them - they must import cleanly.
  log "pip: ConPLex import-chain deps (pysmiles, dscript, deepchem)..."
  pip install -c "${CONSTRAINTS}" pysmiles dscript
  if ! pip install -c "${CONSTRAINTS}" deepchem; then
    warn "deepchem failed to install on this platform."
    warn "ConPLex will still work IF you lazy-import it: move 'import deepchem as dc'"
    warn "in ConPLex/src/featurizers/molecule.py inside the function that uses it."
  fi
fi

if has_model dtkinase || has_model graphban; then
  # dgllife is pure-python but resolves torch/dgl; install after they exist.
  if [ "${INCLUDE_DGL}" = true ] && has_model graphban; then
    log "pip: dgllife (after dgl+torch)..."
    pip install -c "${CONSTRAINTS}" dgllife || warn "dgllife install failed (GraphBAN drug-graph will not build)."
  fi
fi
# NOTE: fair-esm is deliberately NOT pip-installed. dscript may pull it as a
# transitive dep; remove it so the local llm/ESM clone always wins (see step 5).
pip uninstall -y fair-esm 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. graphbolt stub (DGL 2.x dlopen's a version-tagged .so and exit(1)s on
#    mismatch; the models never use graphbolt, so disable it).
# ---------------------------------------------------------------------------
if [ "${INCLUDE_DGL}" = true ]; then
  SP="$(site_packages)"
  GB="${SP}/dgl/graphbolt/__init__.py"
  if [ -f "${GB}" ]; then
    echo "# graphbolt disabled by setup_env_screening.sh - models do not use it" > "${GB}"
    log "Patched graphbolt stub: ${GB}"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Local ESM clone + global sys.path front-insert (.pth executable line)
#    Satisfies DT-Kinase and GraphBAN `import esm` without the segfaulting
#    pip fair-esm, for every interpreter in the env.
# ---------------------------------------------------------------------------
if has_model dtkinase || has_model graphban; then
  if [ "${SCREENING_SKIP_CLONES:-0}" != "1" ]; then
    if [ ! -d "${REPO}/llm/ESM/.git" ] && [ ! -f "${REPO}/llm/ESM/esm/__init__.py" ]; then
      log "Cloning ESM into llm/ESM (pip fair-esm segfaults on mac; clone is canonical)..."
      mkdir -p "${REPO}/llm"
      git clone --depth 1 https://github.com/facebookresearch/esm.git "${REPO}/llm/ESM"
    else
      log "llm/ESM already present."
    fi
  fi
  SP="$(site_packages)"
  PTH="${SP}/zzz_screening_esm.pth"
  # .pth lines starting with 'import ' are executed at interpreter startup.
  echo "import sys; sys.path.insert(0, '${REPO}/llm/ESM')" > "${PTH}"
  log "Registered ESM clone at front of sys.path: ${PTH}"
fi

# ---------------------------------------------------------------------------
# 6. Upstream source clones + patches for the DGL baselines
# ---------------------------------------------------------------------------
if [ "${SCREENING_SKIP_CLONES:-0}" != "1" ]; then
  if has_model drugban; then
    if [ ! -f "${REPO}/DrugBAN/src/dataloader.py" ]; then
      log "Cloning DrugBAN upstream into DrugBAN/src..."
      rm -rf "${REPO}/DrugBAN/src.tmp"
      git clone --depth 1 https://github.com/peizhenbai/DrugBAN.git "${REPO}/DrugBAN/src.tmp"
      # keep repo's existing DrugBAN/src/datasets if present
      mkdir -p "${REPO}/DrugBAN/src"
      cp -R "${REPO}/DrugBAN/src.tmp/." "${REPO}/DrugBAN/src/"
      rm -rf "${REPO}/DrugBAN/src.tmp"
    fi
    # max_drug_nodes 290 -> 310 (kinase molecules up to 310 atoms)
    if [ -f "${REPO}/DrugBAN/src/dataloader.py" ] && grep -q "max_drug_nodes=290" "${REPO}/DrugBAN/src/dataloader.py"; then
      ${PY} - "$REPO/DrugBAN/src/dataloader.py" <<'PYEOF'
import sys
p=sys.argv[1]; s=open(p).read()
open(p,'w').write(s.replace('max_drug_nodes=290','max_drug_nodes=310'))
print(f"  patched max_drug_nodes 290->310 in {p}")
PYEOF
    fi
  fi
  if has_model graphban; then
    if [ ! -f "${REPO}/GraphBAN/src/case_study/models.py" ] && [ ! -f "${REPO}/GraphBAN/src/models.py" ]; then
      log "Cloning GraphBAN upstream into GraphBAN/src..."
      git clone --depth 1 https://github.com/HamidHadipour/GraphBAN.git "${REPO}/GraphBAN/src"
    fi
    if [ -f "${REPO}/GraphBAN/patch_upstream.py" ]; then
      log "Applying GraphBAN upstream patches..."
      ${PY} "${REPO}/GraphBAN/patch_upstream.py" --src "${REPO}/GraphBAN/src" || warn "patch_upstream.py returned non-zero."
    fi
    if [ ! -f "${REPO}/GraphBAN/configs/GraphBAN.yaml" ]; then
      warn "GraphBAN/configs/GraphBAN.yaml is absent - graphban_score.py references it as CANONICAL_CONFIG."
      warn "GraphBAN inference will need that config supplied before it can run."
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 7. Verify (import smoke test for each requested model)
# ---------------------------------------------------------------------------
if [ "${SCREENING_SKIP_VERIFY:-0}" != "1" ]; then
  log "Verifying imports..."
  if ! REPO="${REPO}" MODELS="${MODELS}" INCLUDE_DGL="${INCLUDE_DGL}" ${PY} - <<'PYEOF'
import os, sys, warnings
warnings.filterwarnings("ignore")
repo = os.environ["REPO"]; models = os.environ["MODELS"].split()
ok = True
def check(name, fn):
    global ok
    try:
        fn(); print(f"  OK   {name}")
    except Exception as e:
        ok = False; print(f"  FAIL {name}: {type(e).__name__}: {e}", file=sys.stderr)

import torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu")
print(f"  torch {torch.__version__} | device={dev}")
check("numpy<2",      lambda: __import__("numpy").__version__.startswith("1"))
check("pandas<2.1",   lambda: __import__("pandas"))
check("transformers", lambda: __import__("transformers"))
check("rdkit",        lambda: __import__("rdkit.Chem", fromlist=["Chem"]))

if "dtkinase" in models or "graphban" in models:
    check("esm (local clone)", lambda: __import__("esm"))
if "conplex" in models:
    check("sentencepiece", lambda: __import__("sentencepiece"))
    check("pysmiles",      lambda: __import__("pysmiles"))
    check("dscript",       lambda: __import__("dscript"))
if os.environ["INCLUDE_DGL"] == "true":
    check("dgl", lambda: __import__("dgl"))
    check("dgllife", lambda: __import__("dgllife.utils", fromlist=["smiles_to_bigraph"]))

# score-module import smoke (path hacks + heavy model-code import)
sys.path.insert(0, os.path.join(repo, "scripts", "inference"))
sys.path.insert(0, os.path.join(repo, "scripts", "inference", "models"))
sys.path.insert(0, repo)
score_mods = {"dtkinase":"dtkinase_score","conplex":"conplex_score",
              "drugban":"drugban_score","graphban":"graphban_score"}
import importlib
for m in models:
    mod = score_mods.get(m)
    if mod:
        check(f"{m} score module ({mod})", lambda mod=mod: importlib.import_module(mod))

print("\n  " + ("All checks passed." if ok else "Some checks FAILED - see above."))
sys.exit(0 if ok else 1)
PYEOF
  then warn "Verify smoke test reported failures (see above); env may still be partially usable."; fi
fi

rm -f "${CONSTRAINTS}"
log ""
log "============================================================"
log " Environment '${ENV_NAME}' ready."
log " Activate : conda activate ${ENV_NAME}"
log " Models   : ${MODELS}"
if [ "${PROFILE}" = "mac_arm64" ] && [ "${INCLUDE_DGL}" = true ]; then
  log " NOTE: mac profile uses conda torch 2.3.1 (not 2.4.1) because DGL has"
  log "       no osx-arm64 wheel matched to torch 2.4.1. torch models use MPS."
fi
log " First run downloads model weights (MoLFormer, ProtBert ~1.6GB,"
log "       ESM-1b ~2.5GB for GraphBAN) from HuggingFace / torch.hub."
log "============================================================"
