import os
import time
import random
from huggingface_hub import hf_hub_download

def download_model_files_with_retry(repo_id, filenames, local_dir="./model_files", max_retries=5):
    """Download model files with exponential backoff retry logic."""
    os.makedirs(local_dir, exist_ok=True)
    
    downloaded_files = {}
    
    for filename in filenames:
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to download {filename}")
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    resume_download=True
                )
                downloaded_files[filename] = file_path
                print(f"Successfully downloaded {filename}")
                break
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited. Waiting {wait_time:.2f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Error downloading {filename}: {e}")
                    if attempt == max_retries - 1:
                        print(f"Failed to download {filename} after all retries")
                        downloaded_files[filename] = None
                    else:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Waiting {wait_time:.2f} seconds before retry...")
                        time.sleep(wait_time)
    
    return downloaded_files

if __name__ == "__main__":
    # Files needed for SMI-TED models (Light for testing, Large for production)
    files = [
        "bert_vocab_curated.txt",
        "smi-ted-Light_40.pt",  # ~1.1GB for testing
        "smi_ted_Large.pt"       # ~3GB for production
    ]
    
    print("Downloading SMI-TED model files...")
    print("Note: Downloading both Light (~1.1GB) and Large (~3GB) models")
    downloaded = download_model_files_with_retry(
        repo_id="ibm/materials.smi-ted",
        filenames=files,
        local_dir="./model_files"
    )
    
    print("\nDownload summary:")
    for filename, path in downloaded.items():
        if path:
            print(f"✓ {filename}: {path}")
        else:
            print(f"✗ {filename}: Failed to download")