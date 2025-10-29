# Basic Troubleshooting

**Last Updated**: October 28, 2025  
**Section**: Chapter 01 - Getting Started  
**Audience**: New Users

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Import Errors](#import-errors)
3. [Environment Issues](#environment-issues)
4. [Quick Fixes](#quick-fixes)

---

## Installation Issues

### Python Headers Not Found
**Error**: `fatal error: Python.h: No such file or directory`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install python3.11-dev -y

# macOS
brew install python@3.11

# Then reinstall
python setup.py
```

### RDKit Installation Failed (macOS)
**Error**: `No module named 'rdkit'`

**Solution**: Install via conda for better compatibility:
```bash
conda install -c conda-forge rdkit
```

---

## Import Errors

### ESM Module Not Found
**Error**: `ModuleNotFoundError: No module named 'esm'`

**Solution**:
```bash
source env/bin/activate
pip install fair-esm transformers sentencepiece
```

### PyTorch Not Found
**Error**: `No module named 'torch'`

**Solution**:
```bash
source env/bin/activate
pip install torch torchvision
```

---

## Environment Issues

### Environment Not Activating
**Solution**: Recreate virtual environment
```bash
rm -rf env
python3 -m venv env
source env/bin/activate  # macOS/Linux
# OR env\Scripts\activate  # Windows
python setup.py
```

### CUDA Out of Memory (Linux GPU)
**Solution**: Reduce batch size
```python
# In your configuration
config = {'batch_size': 16}  # Default: 32
```

---

## Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Module not found | `pip install <module>` |
| Permission denied | Use `sudo` or check file permissions |
| Environment issues | Recreate with `python3 -m venv env` |
| Slow downloads | ESM models are large (~3GB), first download takes time |
| GPU not detected | Verify CUDA with `nvidia-smi` |

---

## Getting More Help

- **Detailed Troubleshooting**: [Chapter 07: Troubleshooting](../07-troubleshooting/README.md)
- **Setup Issues**: [Setup Prerequisites](prerequisites.md)
- **Installation Guide**: [Complete Installation Guide](installation.md)

---

**Next Steps**: [User Guide →](../02-user-guide/README.md)
