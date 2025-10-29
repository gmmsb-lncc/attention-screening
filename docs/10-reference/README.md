# Reference

**Last Updated**: October 28, 2025  
**Section**: Chapter 10  
**Audience**: Developers & Advanced Users

---

Technical reference documentation for DockTKinase APIs, configuration, and command-line tools.

## 📚 Contents

### Technical References

1. **[API Reference](api-reference.md)** 🔌
   - Python API documentation
   - Function signatures
   - Class interfaces
   - Usage examples

2. **[Configuration Reference](configuration-reference.md)** ⚙️
   - Configuration file formats
   - Available options
   - Default values
   - Environment variables

3. **[CLI Reference](cli-reference.md)** 💻
   - Command-line interface
   - Available commands
   - Arguments and options
   - Usage examples

4. **[Glossary](glossary.md)** 📖
   - Technical terms
   - Abbreviations
   - Concept definitions

---

## 🎯 Quick Reference

### Common APIs

```python
# Configuration Management
from src.utils.config_manager import ConfigManager
config = ConfigManager(config_path="config.json")

# Device Management
from src.utils.device_manager import DeviceManager
device = DeviceManager.get_device()

# Classification
from src.classifier import Classifier
clf = Classifier(model_name="model_name")
predictions = clf.predict(data)

# Regression
from src.regression import Regressor
reg = Regressor(model_name="model_name")
predictions = reg.predict(data)
```

### Common Commands

```bash
# Run classification pipeline
python run_complete_pipeline.py --mode classify

# Run regression pipeline
python run_regression_pipeline.py

# Run specific model
python -m src.classifier.train --model rf
```

### Configuration Files

- `src/stratification_config.json` - Stratification settings
- `environment.yml` - Conda environment
- `requirements.txt` - Python dependencies

---

## 🔗 Related Documentation

- **User Guide?** → [Chapter 02: User Guide](../02-user-guide/README.md)
- **Modules?** → [Chapter 04: Modules](../04-modules/README.md)
- **Architecture?** → [Chapter 03: Architecture](../03-architecture/README.md)

---

**Previous**: [← Changelogs](../09-changelogs/README.md) | **Home**: [Documentation Index](../README.md)
