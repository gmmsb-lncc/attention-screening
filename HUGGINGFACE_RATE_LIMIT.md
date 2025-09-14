# Handling Hugging Face Rate Limiting

## Problem

When running the DockTKinase pipeline, you may encounter HTTP Error 429 (Too Many Requests) from Hugging Face servers. This happens because the IBM FM4M models try to download required files every time they are loaded, and Hugging Face implements rate limiting to prevent server overload.

## Solution

We've implemented an automated solution that downloads the required model files during environment setup, eliminating the need for repeated downloads:

### Automated Setup (Recommended)

The `setup.sh` script automatically handles everything:
```bash
./setup.sh
```

This will:
1. Create the conda environment
2. Download all required model files
3. Verify the installation

### Manual Setup

If you prefer to set up manually:

1. Run the post-install script after creating the environment:
   ```bash
   cd /home/leon/docktkinase
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

1. Run the setup script to get the model files:
   ```bash
   ./setup.sh
   ```

2. Run your pipeline as usual:
   ```bash
   python docktkinase.py
   ```

## Verification

You can verify that everything is working correctly by running:
```bash
python test_models.py
```

This should output "All tests passed! Models are working correctly."

## Additional Notes

- The downloaded files are stored in `/home/leon/docktkinase/FM4M/model_files/`
- If you need to update the models, simply delete the files in the `model_files` directory and run the post-install script again
- The retry logic will automatically handle temporary rate limiting issues if they occur