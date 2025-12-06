# Glossary

**Last Updated**: December 6, 2025  
**Version**: 2.1

Definitions of technical terms and abbreviations used in DockTKinase.

---

## A-E

### **Boltz-2**
A biomolecular foundation model capable of predicting 3D structures and binding affinities. Used in DockTKinase as an alternative protein embedding strategy.

### **Cross-Attention**
A deep learning mechanism where one sequence (e.g., protein) attends to another (e.g., ligand) to learn interaction patterns.

### **ESM (Evolutionary Scale Modeling)**
A family of protein language models developed by Meta AI (ESM-2) and EvolutionaryScale (ESM-C). They generate numerical representations (embeddings) from protein sequences.

### **Embedding**
A dense vector representation of a biological entity (protein or ligand) that captures its physicochemical and evolutionary properties.

---

## F-J

### **FM4M (Foundation Models for Molecules)**
A suite of models for small molecules. DockTKinase uses **SMI-TED** from this suite to generate ligand embeddings.

### **IC50 (Half Maximal Inhibitory Concentration)**
A measure of the potency of a substance in inhibiting a specific biological or biochemical function. Lower values indicate higher potency.

---

## K-O

### **Kd (Dissociation Constant)**
The equilibrium constant that measures the tendency of a larger object to separate (dissociate) reversibly into smaller components. Lower values indicate higher affinity.

### **Ki (Inhibition Constant)**
An indication of how potent an inhibitor is; it is the concentration required to produce half-maximum inhibition.

### **MPS (Metal Performance Shaders)**
Apple's framework for GPU acceleration on macOS devices (M1/M2/M3 chips). Supported by DockTKinase via PyTorch.

---

## P-T

### **pChEMBL**
A standardized measure of binding affinity, defined as $-\log_{10}(\text{molar value})$.
- $pChEMBL = 9$ corresponds to 1 nM.
- $pChEMBL = 6$ corresponds to 1 $\mu$M.
Used as the target variable for regression.

### **SMILES (Simplified Molecular Input Line Entry System)**
A text notation for describing the structure of chemical molecules using short ASCII strings.

### **Stratification**
The process of splitting data into train/test sets while ensuring that distinct clusters of similar proteins/ligands are kept separate to prevent data leakage.

---

## U-Z

### **ViT (Vision Transformer)**
An architecture originally designed for images, adapted here to treat concatenated protein-ligand sequences as a single input for global context modeling.
