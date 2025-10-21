# ⚡ Quick Start - DockTKinase Execution Order

**Quick guide to execute the pipeline in the correct order.**

## 🚀 **4 Simple Steps**

### **1️⃣ Initial Setup (once only)**
```bash
./setup_conda.sh
```

### **2️⃣ Configure Data**
- Place TSV file in `src/database/`
- Edit `docktkinase.py`:
  ```python
  INPUT_TSV_FILENAME = "your_file.tsv"
  OUTPUT_FOLDER_NAME = "results"
  ```

### **3️⃣ Generate Embeddings**
```bash
conda activate docktkinase
python docktkinase.py
```

### **4️⃣ Train Classifier**
```bash
python run_classifier.py
```

## 📋 **Complete Order**

```bash
# Setup (first time)
./setup_conda.sh

# Activate environment 
conda activate docktkinase

# Complete pipeline
python docktkinase.py      # Embeddings
python run_classifier.py  # Machine Learning
```

## 📚 **Complete Documentation**
- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Detailed guide with all options
- **[src/README.md](src/README.md)** - Modular architecture documentation
- **[README.md](README.md)** - Main project documentation

---
**✨ That's it! In 4 steps you'll have molecular embeddings and trained ML models.**
