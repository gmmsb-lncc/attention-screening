# Handling Hugging Face Rate Limiting

**Data**: 28 de Outubro de 2025  
**Status**: ✅ Resolvido com setup automático

## Problem

When running the DockTKinase pipeline, you may encounter HTTP Error 429 (Too Many Requests) from Hugging Face servers. This happens because the IBM FM4M models try to download required files every time they are loaded, and Hugging Face implements rate limiting to prevent server overload.

## Solution

We've implemented an automated solution that downloads the required model files during environment setup, eliminating the need for repeated downloads:

### Automated Setup (Recommended) ⭐

The **`setup.py`** script automatically handles everything:

```bash
# Setup completo (ambiente + dependências + models)
python setup.py
```

This will:
1. ✅ Create the Python virtual environment (`env/`)
2. ✅ Install all required dependencies
3. ✅ Download all required model files to `./FM4M/model_files/`
4. ✅ Verify the installation
5. ✅ Run validation tests

### Alternative: Post-Install Script

If you already have the environment set up and only need to download models:

```bash
# Apenas baixar modelos FM4M
python scripts/post_install.py
```

This script downloads the necessary files to `./FM4M/model_files/` directory:
- `bert_vocab_curated.txt` - Vocabulary file for tokenization
- `smi-ted-Light_40.pt` - Pre-trained SMI-TED model weights

### Modified Model Loading

The model loading functions have been modified to use local files when available:

```python
def load_smi_ted(folder="./FM4M/model_files", ...):
    # Use local files instead of downloading from Hugging Face
    vocab_path = os.path.join(folder, vocab_filename)
    ckpt_path = os.path.join(folder, ckpt_filename)
    
    # Only download if local files don't exist
    if not os.path.exists(vocab_path):
        # Fallback to downloading
        vocab_path = hf_hub_download(...)
```

### Retry Logic for API Calls

For cases where downloading is still necessary, we've added retry logic with exponential backoff:

```python
def download_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e):
                # Wait with exponential backoff
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise e
```

## Usage

### Complete Setup (First Time)

1. Run the automated setup to get everything:
   ```bash
   # Clone repository
   git clone https://github.com/gmmsb-lncc/docktkinase.git
   cd docktkinase
   
   # Automated setup
   python setup.py
   ```

2. Activate environment and run your pipeline:
   ```bash
   # Activate environment
   source env/bin/activate  # Linux/Mac
   # OR
   env\Scripts\activate  # Windows
   
   # Run classification pipeline
   python run_complete_pipeline.py
   
   # OR run regression pipeline
   python run_regression_pipeline.py --help
   ```

### Quick Start (After Setup)

```bash
# Activate environment
source env/bin/activate

# Run pipeline
python run_complete_pipeline.py \
    --dataset data/test_dataset_1000.tsv \
    --output-dir results/test_run
```

## Verification

You can verify that everything is working correctly by running:

```bash
# Activate environment
source env/bin/activate

# Test imports
python -c "
from FM4M.models import load_smi_ted
print('✅ FM4M models loaded successfully!')
"

# Run validation tests
python tests/test_pipeline_setup.py
```

Expected output:
```
✅ All tests passed! Models are working correctly.
```

## Directory Structure

After successful setup, you should have:

```
docktkinase/
├── env/                           # Virtual environment
├── FM4M/
│   ├── model_files/              # Model files (cached)
│   │   ├── bert_vocab_curated.txt
│   │   └── smi-ted-Light_40.pt
│   └── models/                    # Model code
├── ESM/                          # ESM model (Facebook)
├── src/
│   ├── build/                    # Build pipeline
│   ├── classifier/               # Classification pipeline
│   ├── regression/               # Regression pipeline ⭐ NEW
│   └── utils/                    # Shared utilities ⭐ NEW
├── setup.py                      # Automated setup script
├── run_complete_pipeline.py      # Classification pipeline
└── run_regression_pipeline.py    # Regression pipeline ⭐ NEW
```

## Troubleshooting

### Still Getting 429 Errors?

1. **Check if model files exist**:
   ```bash
   ls -lh FM4M/model_files/
   # Should show:
   # bert_vocab_curated.txt
   # smi-ted-Light_40.pt
   ```

2. **If files are missing, re-run post-install**:
   ```bash
   python scripts/post_install.py
   ```

3. **Check internet connection**:
   - Ensure you can access https://huggingface.co
   - Check firewall settings

4. **Wait and retry**:
   - If you hit rate limit, wait 5-10 minutes
   - The retry logic will automatically wait between attempts

### Model Files Not Found?

```bash
# Verify directory structure
ls -R FM4M/

# Re-download if needed
rm -rf FM4M/model_files/*
python scripts/post_install.py
```

### Environment Issues?

```bash
# Recreate environment
rm -rf env/
python setup.py
```

## Additional Notes

- **Model files location**: `/path/to/docktkinase/FM4M/model_files/`
- **Cache persistence**: Files are cached locally and reused across runs
- **No repeated downloads**: Models downloaded once during setup
- **Automatic fallback**: If local files missing, downloads from HuggingFace
- **Rate limit handling**: Automatic retry with exponential backoff

### Performance Benefits

With local caching:
- ✅ **No network delays** during pipeline execution
- ✅ **Faster startup** (no model downloads)
- ✅ **Offline capability** (after initial setup)
- ✅ **No rate limit errors** (using local files)

## Modern Setup Commands

### Full Installation (Recommended)

```bash
# Complete setup from scratch
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase
python setup.py                    # Creates env + installs deps + downloads models
source env/bin/activate
python run_complete_pipeline.py   # Test classification
python run_regression_pipeline.py --help  # Test regression
```

### Update Existing Installation

```bash
# If you already have env/ but need model updates
cd docktkinase
source env/bin/activate
python scripts/post_install.py    # Re-download models
```

### Clean Reinstall

```bash
# Complete clean reinstall
cd docktkinase
rm -rf env/ FM4M/model_files/
python setup.py                    # Fresh install
```

## System Requirements

- **Python**: 3.8+ (tested with 3.11)
- **Disk Space**: ~5GB for models and dependencies
- **Memory**: 8GB+ RAM recommended
- **Network**: Required for initial setup only

## Related Documentation

- [INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md) - Complete installation guide
- [QUICK_START.md](./QUICK_START.md) - Quick start guide
- [SETUP_PREREQUISITES.md](./SETUP_PREREQUISITES.md) - Prerequisites
- [USER_GUIDE.md](./USER_GUIDE.md) - Complete user manual

---

**Updated**: 28 de Outubro de 2025  
**Status**: ✅ Production Ready  
**Setup Method**: Automated via `setup.py`  
**Dual Pipeline**: Classification + Regression supported
