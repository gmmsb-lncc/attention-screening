# 🔍 Como Verificar a Concatenação de Embeddings - Guia Prático

## Resposta Rápida à Sua Dúvida

```
PERGUNTA: "Ao utilizar o Boltz, o modelo concatena (embedding_dim) 
          esm+boltz+ibm ou concatena apenas boltz+ibm?"

RESPOSTA: ✅ APENAS Boltz + FM4M (IBM é outro nome para FM4M)
         ❌ NÃO concatena ESM (ESM não é usado)

DIMENSÃO: 384 (Boltz) + 768 (FM4M) = 1152 total
```

---

## 1. Verificação Teórica - Olhando o Código

### Arquivo 1: `run_complete_pipeline.py` (Linhas 365-375)

```python
# Determinar dimensão do modelo de proteína (single representation)
protein_dims = {
    # ESM-2 models (mean pooling)
    'esm2_t6_8M_UR50D': 320,
    'esm2_t12_35M_UR50D': 480,
    # ... mais ESM-2
    
    # Boltz-2 (single representation, mean pooling)
    'boltz2': 384  # ← QUANDO VOCÊ USA BOLTZ
}

ligand_dim = 768  # FM4M SMI-TED fixo

# Quando você roda com --protein-model boltz2:
protein_dim = 384
total_dim = ligand_dim + protein_dim  # = 768 + 384 = 1152
```

**Conclusão:** Apenas duas dimensões sendo somadas!

---

### Arquivo 2: `src/build/matrix/embedding_matrix.py` (Linhas 289-305)

```python
def _build_matrix(self) -> np.ndarray:
    """Constrói matriz de embeddings concatenados."""
    
    for entry in self.data:
        ligand_id, seq_id, pki = entry
        
        # Carregar embedding de ligante (FM4M)
        ligand_emb = self._load_embedding(ligand_id)
        # Result: array shape (768,)
        
        # Carregar embedding de proteína (seu modelo escolhido)
        protein_emb = self._load_embedding(seq_id)
        # Result: array shape (384,) se Boltz
        
        # Concatenação simples
        final_embedding = np.concatenate([protein_emb, ligand_emb])
        # Result: array shape (1152,) = 384 + 768
```

**Conclusão:** Apenas dois vetores são concatenados!

---

## 2. Verificação Prática - Rodando o Pipeline

### Passo 1: Executar Pipeline com Boltz

```bash
cd ${PROJECT_ROOT}

python scripts/run_complete_pipeline.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --output results/boltz_test \
    --protein-model boltz2 \
    --device auto \
    --seed 42
```

### Passo 2: Verificar Dimensões dos Arquivos Gerados

```bash
# Verificar matriz final
python << 'EOF'
import numpy as np
from pathlib import Path

matrix_path = Path("results/boltz_test/build/embedding_matrix.npy")
if matrix_path.exists():
    matrix = np.load(matrix_path)
    n_samples, dim = matrix.shape
    
    print("📊 MATRIZ FINAL")
    print(f"   Shape: {matrix.shape}")
    print(f"   Amostras: {n_samples}")
    print(f"   Dimensões: {dim}")
    print()
    
    if dim == 1152:
        print("✅ CORRETO! É Boltz (384) + FM4M (768) = 1152")
    elif dim == 2048:
        print("⚠️  Parece ser ESM-2 t33 (1280) + FM4M (768) = 2048")
    elif dim == 1408:
        print("⚠️  Parece ser ESM-2 t30 (640) + FM4M (768) = 1408")
    else:
        print(f"❓ Dimensão desconhecida: {dim}")
else:
    print("❌ Arquivo não encontrado!")
EOF
```

**Saída esperada:**
```
📊 MATRIZ FINAL
   Shape: (1024, 1152)
   Amostras: 1024
   Dimensões: 1152

✅ CORRETO! É Boltz (384) + FM4M (768) = 1152
```

---

### Passo 3: Verificar Embeddings Individuais

```bash
python << 'EOF'
import numpy as np
from pathlib import Path
import glob

build_dir = Path("results/boltz_test/build/embeddings")

print("🧬 EMBEDDINGS DE PROTEÍNA (Boltz)")
print("=" * 50)

protein_dir = build_dir / "protein_embeddings"
protein_files = list(protein_dir.glob("*.npy"))

if protein_files:
    # Pegar primeiro arquivo
    first_protein = np.load(protein_files[0])
    print(f"   Arquivo: {protein_files[0].name}")
    print(f"   Shape: {first_protein.shape}")
    print(f"   Dtype: {first_protein.dtype}")
    
    if first_protein.shape == (384,):
        print(f"   ✅ Correto! É Boltz-2 (384-dim)")
    else:
        print(f"   ⚠️  Inesperado: esperado (384,)")
else:
    print("   ❌ Nenhum embedding encontrado!")

print()
print("🧬 EMBEDDINGS DE LIGANTE (FM4M)")
print("=" * 50)

ligand_dir = build_dir / "ligand_embeddings"
ligand_files = list(ligand_dir.glob("*.npy"))

if ligand_files:
    # Pegar primeiro arquivo
    first_ligand = np.load(ligand_files[0])
    print(f"   Arquivo: {ligand_files[0].name}")
    print(f"   Shape: {first_ligand.shape}")
    print(f"   Dtype: {first_ligand.dtype}")
    
    if first_ligand.shape == (768,):
        print(f"   ✅ Correto! É FM4M (768-dim)")
    else:
        print(f"   ⚠️  Inesperado: esperado (768,)")
else:
    print("   ❌ Nenhum embedding encontrado!")
EOF
```

**Saída esperada:**
```
🧬 EMBEDDINGS DE PROTEÍNA (Boltz)
==================================================
   Arquivo: PROT_001_protein_embedding.npy
   Shape: (384,)
   Dtype: float32
   ✅ Correto! É Boltz-2 (384-dim)

🧬 EMBEDDINGS DE LIGANTE (FM4M)
==================================================
   Arquivo: LIGAND_001_ligand.npy
   Shape: (768,)
   Dtype: float32
   ✅ Correto! É FM4M (768-dim)
```

---

### Passo 4: Visualizar a Concatenação em Ação

```bash
python << 'EOF'
import numpy as np
from pathlib import Path

# Carregar matriz
matrix = np.load("results/boltz_test/build/embedding_matrix.npy")

# Pegar primeira amostra
sample_1 = matrix[0]

print("🔍 PRIMEIRO VETOR CONCATENADO")
print("=" * 50)
print(f"Shape total: {sample_1.shape} (deve ser (1152,))")
print()

# Separar as partes
boltz_part = sample_1[:384]
fm4m_part = sample_1[384:]

print("📊 ANÁLISE DAS PARTES")
print("=" * 50)
print(f"Boltz (posições 0:384):")
print(f"   Shape: {boltz_part.shape}")
print(f"   Primeiros 5 valores: {boltz_part[:5]}")
print(f"   Média: {boltz_part.mean():.4f}")
print(f"   Std: {boltz_part.std():.4f}")
print()

print(f"FM4M (posições 384:1152):")
print(f"   Shape: {fm4m_part.shape}")
print(f"   Primeiros 5 valores: {fm4m_part[:5]}")
print(f"   Média: {fm4m_part.mean():.4f}")
print(f"   Std: {fm4m_part.std():.4f}")
print()

print("✅ Concatenação correta: 384 + 768 = 1152")
EOF
```

**Saída esperada:**
```
🔍 PRIMEIRO VETOR CONCATENADO
==================================================
Shape total: (1152,) (deve ser (1152,))

📊 ANÁLISE DAS PARTES
==================================================
Boltz (posições 0:384):
   Shape: (384,)
   Primeiros 5 valores: [ 0.234 -0.512  0.145 -0.089  0.723]
   Média: 0.0012
   Std: 0.9845

FM4M (posições 384:1152):
   Shape: (768,)
   Primeiros 5 valores: [-0.156  0.789 -0.234  0.567 -0.345]
   Média: -0.0034
   Std: 1.0234

✅ Concatenação correta: 384 + 768 = 1152
```

---

## 3. Comparação - Boltz vs ESM-2

### Teste: Rode Com Diferentes Modelos

```bash
# Teste 1: COM BOLTZ
python scripts/run_complete_pipeline.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --output results/test_boltz \
    --protein-model boltz2

# Teste 2: COM ESM-2 t33
python scripts/run_complete_pipeline.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --output results/test_esm2_t33 \
    --protein-model esm2_t33_650M_UR50D
```

### Verificar Diferença

```bash
python << 'EOF'
import numpy as np

boltz_matrix = np.load("results/test_boltz/build/embedding_matrix.npy")
esm2_matrix = np.load("results/test_esm2_t33/build/embedding_matrix.npy")

print("📊 COMPARAÇÃO DE DIMENSÕES")
print("=" * 50)
print(f"Boltz:      {boltz_matrix.shape}")
print(f"ESM-2 t33:  {esm2_matrix.shape}")
print()

boltz_dim = boltz_matrix.shape[1]
esm2_dim = esm2_matrix.shape[1]

print("🔢 ANÁLISE")
print("=" * 50)
print(f"Boltz dimensões: {boltz_dim}")
print(f"  = 384 (Boltz) + 768 (FM4M)")
print()
print(f"ESM-2 dimensões: {esm2_dim}")
print(f"  = 1280 (ESM-2) + 768 (FM4M)")
print()

if boltz_dim == 1152 and esm2_dim == 2048:
    print("✅ CONFIRMADO!")
    print("   Boltz usa APENAS Boltz + FM4M")
    print("   ESM-2 usa APENAS ESM-2 + FM4M")
    print("   (Nunca há ESM + Boltz + FM4M)")
EOF
```

**Saída esperada:**
```
📊 COMPARAÇÃO DE DIMENSÕES
==================================================
Boltz:      (1024, 1152)
ESM-2 t33:  (1024, 2048)

🔢 ANÁLISE
==================================================
Boltz dimensões: 1152
  = 384 (Boltz) + 768 (FM4M)

ESM-2 dimensões: 2048
  = 1280 (ESM-2) + 768 (FM4M)

✅ CONFIRMADO!
   Boltz usa APENAS Boltz + FM4M
   ESM-2 usa APENAS ESM-2 + FM4M
   (Nunca há ESM + Boltz + FM4M)
```

---

## 4. Verificação no Log do Pipeline

Ao rodar o pipeline, observe a saída:

```bash
python scripts/run_complete_pipeline.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --protein-model boltz2 2>&1 | grep -E "(Protein Model|Total Embedding|embedding_dim)"
```

**Saída esperada:**
```
   Protein Model: boltz2 (384-dim padrão)
   Total Embedding: 1152-dim (Ligand: 768 + Protein: 384)
```

---

## 5. Script Completo de Verificação

```bash
#!/bin/bash
# verify_embeddings.sh

RESULTS_DIR="results/boltz_test"

echo "🔍 VERIFICAÇÃO COMPLETA DE EMBEDDINGS"
echo "======================================"
echo

# 1. Matriz final
echo "1️⃣  MATRIZ CONCATENADA"
python << 'EOF'
import numpy as np
m = np.load(f"{RESULTS_DIR}/build/embedding_matrix.npy")
print(f"   Shape: {m.shape}")
print(f"   Total dimensões: {m.shape[1]}")
if m.shape[1] == 1152:
    print("   ✅ Boltz (384) + FM4M (768) = 1152")
EOF
echo

# 2. Embeddings individuais
echo "2️⃣  EMBEDDINGS DE PROTEÍNA"
python << 'EOF'
import numpy as np
from pathlib import Path
pfile = list(Path(f"{RESULTS_DIR}/build/embeddings/protein_embeddings/").glob("*.npy"))[0]
p = np.load(pfile)
print(f"   Arquivo: {pfile.name}")
print(f"   Shape: {p.shape}")
if p.shape == (384,):
    print("   ✅ Boltz-2")
EOF
echo

# 3. Ligantes
echo "3️⃣  EMBEDDINGS DE LIGANTE"
python << 'EOF'
import numpy as np
from pathlib import Path
lfile = list(Path(f"{RESULTS_DIR}/build/embeddings/ligand_embeddings/").glob("*.npy"))[0]
l = np.load(lfile)
print(f"   Arquivo: {lfile.name}")
print(f"   Shape: {l.shape}")
if l.shape == (768,):
    print("   ✅ FM4M")
EOF
echo

echo "======================================"
echo "✅ VERIFICAÇÃO COMPLETA"
```

---

## 📋 Checklist Final

- [ ] Matriz final tem 1152 dimensões? ✅
- [ ] Embeddings de proteína têm 384 dimensões? ✅
- [ ] Embeddings de ligante têm 768 dimensões? ✅
- [ ] Log do pipeline mostra "384-dim + 768-dim = 1152-dim"? ✅
- [ ] Nenhuma menção a ESM no modelo de proteína? ✅

Se tudo está ✅, então está correto: **Apenas Boltz + FM4M!**

---

## 🎓 Conclusão

```
Boltz + FM4M concatenation verificado!

❌ NÃO CONCATENA: ESM + Boltz + FM4M (nunca!)
✅ CONCATENA: Boltz (384) + FM4M (768) = 1152

Cada modelo de proteína é exclusivo:
- Escolher Boltz = NÃO usa ESM
- Escolher ESM-2 = NÃO usa Boltz
```

