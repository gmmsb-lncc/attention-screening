#!/usr/bin/env python3
"""
Download Boltz CCD data from HuggingFace
"""
import requests
from pathlib import Path
from tqdm import tqdm
import sys

cache_dir = Path.home() / '.boltz'
cache_dir.mkdir(parents=True, exist_ok=True)
tar_path = cache_dir / 'mols.tar'

if tar_path.exists():
    print(f'Removing existing incomplete file: {tar_path}')
    tar_path.unlink()

MOL_URL = 'https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar'
print(f'Downloading CCD data from HuggingFace...')
print(f'URL: {MOL_URL}')
print('Size: ~1.8GB - this may take 10-20 minutes on slow connection...')
print()

try:
    response = requests.get(MOL_URL, stream=True, timeout=120)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    print(f'Total size: {total_size / (1024*1024*1024):.2f} GB')
    
    with open(tar_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc='Downloading'
    ) as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print()
    print(f'✓ Download completed: {tar_path}')
    final_size = tar_path.stat().st_size
    print(f'  Size: {final_size / (1024*1024*1024):.2f} GB')
    
    if final_size < total_size * 0.99:
        print(f'⚠ Warning: Downloaded size ({final_size}) is less than expected ({total_size})')
        sys.exit(1)
    
    print('\nNow extracting archive...')
    import tarfile
    with tarfile.open(tar_path) as tar:
        tar.extractall(cache_dir)
    
    print(f'✓ Extraction completed to {cache_dir}')
    print('\nBoltz CCD data ready!')
    
except Exception as e:
    print(f'\n❌ Failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
