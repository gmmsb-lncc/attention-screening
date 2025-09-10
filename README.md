# DockTKinase

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 🧬 Overview

DockTKinase is a computational pipeline for generating molecular embeddings of kinase inhibitors and their target proteins, specifically designed for non-human kinases. The pipeline leverages IBM's Foundation Models for Materials (FM4M) to create high-quality representations that can be used for various downstream tasks such as drug discovery, virtual screening, and structure-activity relationship studies.

This tool is particularly valuable for researchers working on neglected tropical diseases, veterinary medicine, or comparative studies between human and non-human kinases, where traditional drug discovery approaches may be limited by data availability.

## 🚀 Key Features

- **Automated Pipeline**: End-to-end processing of kinase-compound interactions
- **Multi-Modal Embeddings**: Generates embeddings for both ligands (small molecules) and proteins (kinases)
- **Checkpoint System**: Resumable processing with automatic checkpoint management
- **Scalable Processing**: Uses Apache Spark for efficient large-scale computations
- **Foundation Model Integration**: Leverages IBM's FM4M models for state-of-the-art representations
- **Specialized for Non-Human Kinases**: Focused on kinases from pathogens and model organisms

## 📁 Project Structure

```
docktkinase/
├── docktkinase.py              # Main entry point and configuration
├── src/
│   ├── database/               # Input data (TSV files)
│   ├── build/                  # Pipeline core implementation
│   │   ├── embeddingPreparation.py
│   │   ├── embeddingBuild.py
│   │   ├── embeddingIBM.py
│   │   ├── embeddingMeta.py
│   │   ├── buildEmbeddingMain.py
│   ├── interface.py            # Pipeline interface and execution manager
├── materials/                  # IBM FM4M models and dependencies
├── non_human/                  # Default output directory
├── environment.yml             # Conda environment specification
├── LICENSE
└── README.md
```

## 🧪 Input Data Format

The pipeline expects input data in TSV format with the following columns:

| Column | Description |
|--------|-------------|
| `chembl_id` | ChEMBL identifier for the compound-target pair |
| `molregno` | Molecule registration number in ChEMBL |
| `target_kinase` | Name of the kinase target |
| `canonical_smiles` | Canonical SMILES representation of the compound |
| `standard_value` | Activity value (e.g., IC50, Ki) |
| `standard_type` | Type of activity measurement |
| `pchembl_value` | Negative logarithm of the activity value |
| `compound_name` | Common name of the compound |
| `organism` | Organism of the kinase target |
| `seq` | Protein sequence of the kinase |
| `seq_id` | Unique identifier for the protein sequence |

## ⚙️ Installation

### Prerequisites
- Conda or Miniconda
- Git

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/docktkinase.git
   cd docktkinase
   ```

2. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate docktkinase
   ```

3. **Prepare input data**:
   Place your TSV file in `src/database/` with the appropriate format.

## ▶️ Usage

### Configuration

Edit `docktkinase.py` to set your input file and output directory:

```python
# Input TSV filename (must be in src/database/)
INPUT_TSV_FILENAME = "kinase_non_human_compounds.tsv"

# Output folder name
OUTPUT_FOLDER_NAME = "non_human"
```

### Execution

Run the pipeline:
```bash
python docktkinase.py
```

The pipeline execution follows these stages:
1. **Data Preparation**: Processes the input TSV file to extract unique ligands and proteins
2. **Ligand Embedding Generation**: Creates embeddings for ligands using IBM's SMI-TED model
3. **Protein Embedding Generation**: Creates embeddings for proteins using Meta's ESM model
4. **Matrix Construction**: Combines embeddings into matrices for downstream analysis

## 📊 Output Structure

The pipeline generates the following outputs in the specified output directory:

```
output_folder/
├── ligand/                     # Individual ligand SMILES files
├── protein/                    # Individual protein FASTA files
├── ligand_embeddings/          # Generated ligand embeddings (NumPy arrays)
├── protein_embeddings/         # Generated protein embeddings (NumPy arrays)
├── matrix_embedding/           # Combined embedding matrices:
│   ├── ligand_matrix_cls.npy   # Ligand embeddings (CLS tokens)
│   ├── ligand_matrix_mean.npy  # Ligand embeddings (mean pooling)
│   ├── protein_matrix_cls.npy  # Protein embeddings (CLS tokens)
│   └── protein_matrix_mean.npy # Protein embeddings (mean pooling)
├── unique_ligands.csv          # Processed unique ligands
├── unique_proteins.csv         # Processed unique proteins
└── embedding_checkpoint.txt    # Pipeline execution checkpoint
```

## 🛠️ Advanced Configuration

### Environment Settings

Key configuration options in `docktkinase.py`:
- `INPUT_TSV_FILENAME`: Input TSV file name (must be in `src/database/`)
- `OUTPUT_FOLDER_NAME`: Output directory name
- The pipeline automatically uses the docktkinase conda environment

### Spark Configuration

For large datasets, adjust Spark settings in `src/build/embeddingBuild.py`:
- Memory allocation
- Number of cores
- Partitioning strategy

## 🔧 Troubleshooting

### Common Issues

1. **Module Not Found Errors**
   - Ensure you're using the correct conda environment: `conda activate docktkinase`
   - Verify all dependencies are installed: `conda list`

2. **Empty Embedding Directories**
   - Check that input data contains valid SMILES and protein sequences
   - Verify the TSV file format matches the expected schema
   - Clear checkpoints if resuming from a corrupted state

3. **Memory Issues**
   - Reduce batch sizes in embedding generation scripts
   - Adjust Spark configuration based on available system resources
   - Process smaller subsets of data

4. **Checkpoint Problems**
   - Delete `embedding_checkpoint.txt` to restart the pipeline from scratch
   - Check file permissions on output directories

5. **Hugging Face Rate Limiting (HTTP 429 Errors)**
   - See [HUGGINGFACE_RATE_LIMIT.md](HUGGINGFACE_RATE_LIMIT.md) for detailed instructions
   - Pre-download model files to avoid repeated downloads
   - Use local model files when available

### Debugging

Enable verbose logging by modifying the log level in `embeddingBuild.py`:
```python
# Change this line to increase verbosity
sc.setLogLevel("INFO")  # or "DEBUG"
```

## 📚 Related Technologies

This project integrates several cutting-edge technologies:

- [IBM Foundation Models for Materials (FM4M)](https://github.com/IBM/materials) - State-of-the-art foundation models for molecular representations
- [ChEMBL Database](https://www.ebi.ac.uk/chembl/) - Manually curated database of bioactive molecules
- [RDKit](https://www.rdkit.org/) - Open-source cheminformatics software
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Apache Spark](https://spark.apache.org/) - Distributed computing engine

## 📈 Applications

DockTKinase is particularly useful for:

1. **Drug Discovery for Neglected Diseases**: Identifying potential therapeutics for pathogens
2. **Comparative Kinase Studies**: Understanding differences between human and pathogen kinases
3. **Virtual Screening**: Rapid identification of potential lead compounds
4. **Structure-Activity Relationship (SAR) Analysis**: Understanding molecular determinants of activity
5. **Polypharmacology Studies**: Investigating compound promiscuity across kinase families

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions from the community! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- IBM Research for providing the Foundation Models for Materials
- The ChEMBL team for maintaining the comprehensive database of bioactive molecules
- The open-source community for the tools and libraries that make this project possible
- Contributors to the RDKit, PyTorch, and Apache Spark projects

## 📞 Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the maintainers directly.