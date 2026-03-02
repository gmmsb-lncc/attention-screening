# Level 5-Lite: Cross-Attention com Embeddings Pré-calculados

## 🎯 Objetivo: MCC > 0.50 (Target: 0.52-0.55)

Este documento descreve a arquitetura **Level 5-Lite**, uma versão simplificada e pragmática do Level 5 que:
- Elimina a complexidade do alinhamento SMILES → átomos
- Aproveita embeddings **já calculados** (MoLFormer + ESM-2)
- Testa a hipótese central: **cross-attention bidirecional melhora a predição?**

---

## 📋 Sumário

1. [Motivação e Justificativa](#1-motivação-e-justificativa)
2. [Arquitetura Proposta](#2-arquitetura-proposta)
3. [Componentes Detalhados](#3-componentes-detalhados)
4. [Dados de Entrada](#4-dados-de-entrada)
5. [Implementação Passo-a-Passo](#5-implementação-passo-a-passo)
6. [Integração com CLI](#6-integração-com-cli)
7. [Hiperparâmetros](#7-hiperparâmetros)
8. [Métricas e Avaliação](#8-métricas-e-avaliação)
9. [Checklist de Implementação](#9-checklist-de-implementação)

---

## 1. Motivação e Justificativa

### 1.1 Contexto: Resultados Atuais

| Level | Arquitetura | MCC (human, scaffold) |
|-------|-------------|----------------------|
| **Level 1** | FP + MLP | **0.428** ← melhor atual |
| Level 2 | Emb + MLP | 0.390 |
| Level 3 | CNN | < 0.428 (inferior) |
| Level 4 | CNN + CA | ~0.45 (estimado) |

**Problema:** Arquiteturas mais complexas (Level 3/4) não superam o baseline simples (Level 1).

### 1.2 Hipótese a Testar

> **Hipótese:** O gargalo dos Levels 3/4 é a representação de entrada (matrizes per-token com CNN), não a falta de cross-attention. Usar **vetores mean-pooled** com **Transformer + Cross-Attention** pode ser mais eficiente.

### 1.3 Por que Level 5-Lite (não Level 5-Full)?

```
┌─────────────────────────────────────────────────────────────────┐
│                 LEVEL 5-FULL vs LEVEL 5-LITE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 5-Full (GNN + Transformer):                              │
│  ─────────────────────────────────                              │
│  ✗ Requer alinhamento SMILES → átomos (2-3 semanas)             │
│  ✗ GNN do zero ou híbrido com MoLFormer                         │
│  ✗ Risco técnico alto                                           │
│  ✗ Debugging complexo                                           │
│                                                                 │
│  Level 5-Lite (Transformer + Cross-Attention):                  │
│  ─────────────────────────────────────────────                  │
│  ✓ Usa matrizes já calculadas (sem alinhamento)                 │
│  ✓ Implementação em 3-5 dias                                    │
│  ✓ Risco técnico baixo                                          │
│  ✓ Testa hipótese central rapidamente                           │
│  ✓ Se funcionar, upgrade para GNN depois                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Fundamentação Científica

#### **Cross-Attention Bidirecional**

A interação proteína-ligante é **bidirecional**:

```
Proteína "vê" o ligante:
• Resíduos do binding pocket reconhecem grupos químicos
• Modelado por: CrossAttn(Query=Protein, Key/Value=Ligand)

Ligante "vê" a proteína:
• Átomos do ligante interagem com resíduos específicos
• Modelado por: CrossAttn(Query=Ligand, Key/Value=Protein)
```

Papers que validam esta abordagem:
- **MolTrans** (Huang et al., 2021): MCC +5% com cross-attention
- **TargetFormer** (Zhang et al., 2023): MCC = 0.59 com cross-attention bidirecional

#### **Attention Pooling vs Mean Pooling**

```
Mean Pooling (Level 2):
• Todos os tokens têm peso igual
• Perde informação de "quais tokens são importantes"
• Simples mas subótimo

Attention Pooling (Level 5-Lite):
• Learnable query "pergunta" quais tokens importam
• Binding pocket residues recebem mais peso automaticamente
• Farmacóforos do ligante recebem mais peso
• Ganho esperado: +2-3% MCC
```

---

## 2. Arquitetura Proposta

### 2.1 Visão Geral

```
┌─────────────────────────────────────────────────────────────────┐
│                      LEVEL 5-LITE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PROTEIN INPUT                     LIGAND INPUT                 │
│  ──────────────                    ────────────                 │
│  ESM-2 matrix                      MoLFormer matrix             │
│  [L, 320]                          [T, 768]                     │
│  (per-residue)                     (per-token)                  │
│       ↓                                 ↓                       │
│  Linear Projection                 Linear Projection            │
│  [L, 320] → [L, 512]               [T, 768] → [T, 512]          │
│       ↓                                 ↓                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           TRANSFORMER ENCODER (shared dim=512)           │   │
│  │                                                          │   │
│  │  Protein Branch:              Ligand Branch:             │   │
│  │  2x Transformer layers        2x Transformer layers      │   │
│  │  (self-attention)             (self-attention)           │   │
│  │       ↓                            ↓                     │   │
│  │  [L, 512]                     [T, 512]                   │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                 ↓                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           CROSS-ATTENTION BLOCK (bidirecional)           │   │
│  │                                                          │   │
│  │  Protein → Ligand:                                       │   │
│  │    Query: protein [L, 512]                               │   │
│  │    Key/Value: ligand [T, 512]                            │   │
│  │    Output: protein_cross [L, 512]                        │   │
│  │                                                          │   │
│  │  Ligand → Protein:                                       │   │
│  │    Query: ligand [T, 512]                                │   │
│  │    Key/Value: protein [L, 512]                           │   │
│  │    Output: ligand_cross [T, 512]                         │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                 ↓                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ATTENTION POOLING                           │   │
│  │                                                          │   │
│  │  Protein: learnable query [1, 512]                       │   │
│  │    → AttentionPool(protein_cross) → [512]                │   │
│  │                                                          │   │
│  │  Ligand: learnable query [1, 512]                        │   │
│  │    → AttentionPool(ligand_cross) → [512]                 │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                 ↓                       │
│  [512]                             [512]                        │
│        └────────────┬─────────────┘                             │
│                     ↓                                           │
│              Concatenate [1024]                                 │
│                     ↓                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CLASSIFIER HEAD                             │   │
│  │                                                          │   │
│  │  Linear(1024, 512) → GELU → LayerNorm → Dropout(0.3)     │   │
│  │  Linear(512, 256) → GELU → LayerNorm → Dropout(0.3)      │   │
│  │  Linear(256, 1)                                          │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                     ↓                                           │
│              Sigmoid → P(active)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Dimensões

| Componente | Input Shape | Output Shape |
|------------|-------------|--------------|
| Protein Matrix (ESM-2) | `[L, 320]` | — |
| Ligand Matrix (MoLFormer) | `[T, 768]` | — |
| Protein Projection | `[L, 320]` | `[L, 512]` |
| Ligand Projection | `[T, 768]` | `[T, 512]` |
| Transformer Encoder | `[*, 512]` | `[*, 512]` |
| Cross-Attention | `[*, 512]` | `[*, 512]` |
| Attention Pool | `[*, 512]` | `[512]` |
| Concat | `[512] + [512]` | `[1024]` |
| Classifier | `[1024]` | `[1]` |

Onde:
- `L` = comprimento da sequência proteica (variável, max 1024)
- `T` = número de tokens SMILES (variável, tipicamente 10-100)

---

## 3. Componentes Detalhados

### 3.1 Protein Encoder

```python
class ProteinEncoder(nn.Module):
    """
    Encoder para embeddings ESM-2 per-residue.
    
    ESM-2 já codifica informação evolutiva e estrutural.
    O Transformer encoder refina para a tarefa de binding.
    
    Justificativa científica:
    - ESM-2 foi treinado em 250M+ sequências (conhecimento geral)
    - Transformer encoder especializa para binding (fine-tuning leve)
    - Self-attention captura dependências de longo alcance na sequência
    """
    
    def __init__(
        self,
        input_dim: int = 320,      # ESM-2 8M hidden dim
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Projeção linear para dimensão uniforme
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN (mais estável)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: [batch, seq_len, 320] ESM-2 embeddings
            mask: [batch, seq_len] padding mask (True = pad)
        
        Returns:
            [batch, seq_len, 512]
        """
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return x
```

### 3.2 Ligand Encoder

```python
class LigandEncoder(nn.Module):
    """
    Encoder para embeddings MoLFormer per-token.
    
    MoLFormer já codifica química molecular (1.1B parâmetros).
    O Transformer encoder refina para a tarefa de binding.
    
    Justificativa científica:
    - MoLFormer foi treinado em 2M+ moléculas (conhecimento químico)
    - Self-attention entre tokens SMILES captura dependências locais
    - Não precisa de GNN porque MoLFormer já entende estrutura
    """
    
    def __init__(
        self,
        input_dim: int = 768,      # MoLFormer hidden dim
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: [batch, n_tokens, 768] MoLFormer embeddings
            mask: [batch, n_tokens] padding mask
        
        Returns:
            [batch, n_tokens, 512]
        """
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return x
```

### 3.3 Bidirectional Cross-Attention

```python
class BidirectionalCrossAttention(nn.Module):
    """
    Cross-attention bidirecional entre proteína e ligante.
    
    Justificativa científica:
    - Proteína → Ligante: quais grupos químicos o binding pocket "vê"
    - Ligante → Proteína: quais resíduos o farmacóforo "vê"
    - Bidirecional captura a complementaridade da interação
    
    Referência: TargetFormer (Zhang et al., Nature Comm 2023)
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Protein queries ligand
        self.protein_to_ligand = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Ligand queries protein
        self.ligand_to_protein = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms (Pre-LN style)
        self.norm_p = nn.LayerNorm(hidden_dim)
        self.norm_l = nn.LayerNorm(hidden_dim)
        
        # Feed-forward após cross-attention
        self.ffn_p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_l = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.norm_ffn_p = nn.LayerNorm(hidden_dim)
        self.norm_ffn_l = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        protein: torch.Tensor,      # [B, L, D]
        ligand: torch.Tensor,       # [B, T, D]
        protein_mask: torch.Tensor = None,  # [B, L]
        ligand_mask: torch.Tensor = None,   # [B, T]
    ):
        """
        Returns:
            protein_out: [B, L, D] - protein enriched with ligand info
            ligand_out: [B, T, D] - ligand enriched with protein info
        """
        # Protein attends to ligand
        p_norm = self.norm_p(protein)
        p_cross, _ = self.protein_to_ligand(
            query=p_norm,
            key=ligand,
            value=ligand,
            key_padding_mask=ligand_mask,
        )
        protein = protein + p_cross  # Residual
        protein = protein + self.ffn_p(self.norm_ffn_p(protein))
        
        # Ligand attends to protein
        l_norm = self.norm_l(ligand)
        l_cross, _ = self.ligand_to_protein(
            query=l_norm,
            key=protein,
            value=protein,
            key_padding_mask=protein_mask,
        )
        ligand = ligand + l_cross  # Residual
        ligand = ligand + self.ffn_l(self.norm_ffn_l(ligand))
        
        return protein, ligand
```

### 3.4 Attention Pooling

```python
class AttentionPooling(nn.Module):
    """
    Pooling com query aprendível.
    
    Justificativa científica:
    - Mean pooling trata todos os tokens igualmente (subótimo)
    - Attention pooling aprende quais tokens são importantes
    - Para proteínas: binding pocket residues recebem mais peso
    - Para ligantes: farmacóforos recebem mais peso
    
    Referência: Set Transformer (Lee et al., ICML 2019)
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Learnable query (1 token que "pergunta" o resumo)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] padding mask
        
        Returns:
            [batch, hidden_dim]
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # Attention pooling
        pooled, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=mask,
        )
        
        pooled = self.norm(pooled)
        
        return pooled.squeeze(1)  # [B, D]
```

### 3.5 Classifier Head

```python
class ClassifierHead(nn.Module):
    """
    MLP para classificação binária.
    
    Arquitetura com regularização forte:
    - LayerNorm após cada camada (estabilidade)
    - Dropout 0.3 (previne overfitting)
    - GELU activation (melhor que ReLU para transformers)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # protein_dim + ligand_dim
        hidden_dims: list = [512, 256],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, input_dim]
        
        Returns:
            [batch, 1] logits
        """
        return self.classifier(x)
```

### 3.6 Modelo Completo

```python
class Level5LiteModel(nn.Module):
    """
    Level 5-Lite: Cross-Attention com Embeddings Pré-calculados.
    
    Arquitetura:
    1. Protein Encoder (ESM-2 matrices → Transformer)
    2. Ligand Encoder (MoLFormer matrices → Transformer)
    3. Bidirectional Cross-Attention
    4. Attention Pooling
    5. Classifier Head
    """
    
    def __init__(
        self,
        protein_input_dim: int = 320,
        ligand_input_dim: int = 768,
        hidden_dim: int = 512,
        num_encoder_layers: int = 2,
        num_cross_attn_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        
        # Encoders
        self.protein_encoder = ProteinEncoder(
            input_dim=protein_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.ligand_encoder = LigandEncoder(
            input_dim=ligand_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_cross_attn_layers)
        ])
        
        # Attention pooling
        self.protein_pool = AttentionPooling(hidden_dim, num_heads, dropout)
        self.ligand_pool = AttentionPooling(hidden_dim, num_heads, dropout)
        
        # Classifier
        self.classifier = ClassifierHead(
            input_dim=hidden_dim * 2,  # concat protein + ligand
            hidden_dims=[hidden_dim, hidden_dim // 2],
            dropout=classifier_dropout,
        )
        
    def forward(
        self,
        protein_matrix: torch.Tensor,   # [B, L, 320]
        ligand_matrix: torch.Tensor,    # [B, T, 768]
        protein_mask: torch.Tensor = None,
        ligand_mask: torch.Tensor = None,
    ):
        """
        Args:
            protein_matrix: ESM-2 per-residue embeddings
            ligand_matrix: MoLFormer per-token embeddings
            protein_mask: Padding mask for protein (True = pad)
            ligand_mask: Padding mask for ligand (True = pad)
        
        Returns:
            logits: [batch, 1]
        """
        # Encode
        protein = self.protein_encoder(protein_matrix, protein_mask)
        ligand = self.ligand_encoder(ligand_matrix, ligand_mask)
        
        # Cross-attention
        for cross_attn in self.cross_attn_layers:
            protein, ligand = cross_attn(
                protein, ligand, protein_mask, ligand_mask
            )
        
        # Pool to fixed-size vectors
        protein_vec = self.protein_pool(protein, protein_mask)  # [B, 512]
        ligand_vec = self.ligand_pool(ligand, ligand_mask)      # [B, 512]
        
        # Classify
        combined = torch.cat([protein_vec, ligand_vec], dim=-1)  # [B, 1024]
        logits = self.classifier(combined)
        
        return logits
```

---

## 4. Dados de Entrada

### 4.1 Estrutura de Arquivos

```
results/protein_model_benchmark_{human|non_human}_v2/
└── esm2_t6_8M_UR50D/
    └── build/
        ├── protein_matrices/
        │   ├── {seq_id}_matrix.npy      # [L, 320] per-residue ESM-2
        │   └── ...
        ├── molformer_matrix/
        │   ├── {chembl_id}_molformer_matrix.npy  # [T, 768] per-token MoLFormer
        │   └── ...
        └── ligand_embeddings/
            ├── {chembl_id}_molformer_embedding.npy  # [768] mean-pooled (Level 2)
            └── ...
```

### 4.2 Scaffold Splits

```
scaffolds_splits/output/
├── human_test.tsv.gz           # 40,471 linhas
├── non_human_test.tsv.gz
└── scenarios/
    └── Sc/
        ├── human_train.tsv.gz  # 269,716 linhas
        ├── human_val.tsv.gz    # 65,169 linhas
        ├── non_human_train.tsv.gz
        └── non_human_val.tsv.gz
```

### 4.3 Formato das Matrizes

| Matriz | Shape | Dtype | Descrição |
|--------|-------|-------|-----------|
| Protein (ESM-2) | `[L, 320]` | float32 | L = seq_len (max 1024) |
| Ligand (MoLFormer) | `[T, 768]` | float32 | T = n_tokens SMILES |

### 4.4 Dataset Class

```python
class Level5LiteDataset(torch.utils.data.Dataset):
    """
    Dataset para Level 5-Lite.
    
    Carrega:
    - Protein matrix: ESM-2 per-residue [L, 320]
    - Ligand matrix: MoLFormer per-token [T, 768]
    - Label: binary (pChEMBL >= 6.0 → active)
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        protein_matrix_dir: str,
        ligand_matrix_dir: str,
        max_protein_len: int = 1024,
        max_ligand_len: int = 256,
    ):
        self.data = data_df
        self.protein_dir = protein_matrix_dir
        self.ligand_dir = ligand_matrix_dir
        self.max_protein_len = max_protein_len
        self.max_ligand_len = max_ligand_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load matrices
        protein_path = os.path.join(
            self.protein_dir, f"{row['seq_id']}_matrix.npy"
        )
        ligand_path = os.path.join(
            self.ligand_dir, f"{row['chembl_id']}_molformer_matrix.npy"
        )
        
        protein_matrix = np.load(protein_path)  # [L, 320]
        ligand_matrix = np.load(ligand_path)    # [T, 768]
        
        # Truncate if needed
        protein_matrix = protein_matrix[:self.max_protein_len]
        ligand_matrix = ligand_matrix[:self.max_ligand_len]
        
        # Label
        label = 1 if row['pchembl_value'] >= 6.0 else 0
        
        return {
            'protein_matrix': torch.tensor(protein_matrix, dtype=torch.float32),
            'ligand_matrix': torch.tensor(ligand_matrix, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'pchembl': torch.tensor(row['pchembl_value'], dtype=torch.float32),
        }


def collate_level5_lite(batch):
    """
    Collate function com padding dinâmico.
    """
    # Find max lengths in batch
    max_protein_len = max(b['protein_matrix'].size(0) for b in batch)
    max_ligand_len = max(b['ligand_matrix'].size(0) for b in batch)
    
    protein_matrices = []
    ligand_matrices = []
    protein_masks = []
    ligand_masks = []
    labels = []
    pchembls = []
    
    for b in batch:
        # Pad protein
        p = b['protein_matrix']
        p_len = p.size(0)
        p_padded = F.pad(p, (0, 0, 0, max_protein_len - p_len))
        protein_matrices.append(p_padded)
        protein_masks.append(
            torch.cat([
                torch.zeros(p_len, dtype=torch.bool),
                torch.ones(max_protein_len - p_len, dtype=torch.bool)
            ])
        )
        
        # Pad ligand
        l = b['ligand_matrix']
        l_len = l.size(0)
        l_padded = F.pad(l, (0, 0, 0, max_ligand_len - l_len))
        ligand_matrices.append(l_padded)
        ligand_masks.append(
            torch.cat([
                torch.zeros(l_len, dtype=torch.bool),
                torch.ones(max_ligand_len - l_len, dtype=torch.bool)
            ])
        )
        
        labels.append(b['label'])
        pchembls.append(b['pchembl'])
    
    return {
        'protein_matrix': torch.stack(protein_matrices),
        'ligand_matrix': torch.stack(ligand_matrices),
        'protein_mask': torch.stack(protein_masks),
        'ligand_mask': torch.stack(ligand_masks),
        'label': torch.stack(labels),
        'pchembl': torch.stack(pchembls),
    }
```

---

## 5. Implementação Passo-a-Passo

### Passo 1: Criar Módulo `level5_lite/`

```bash
mkdir -p crossattention_split_analysis/models/level5_lite/
touch crossattention_split_analysis/models/level5_lite/__init__.py
```

Arquivos a criar:

```
crossattention_split_analysis/models/level5_lite/
├── __init__.py
├── model.py          # Level5LiteModel
├── encoders.py       # ProteinEncoder, LigandEncoder
├── attention.py      # BidirectionalCrossAttention, AttentionPooling
├── classifier.py     # ClassifierHead
└── dataset.py        # Level5LiteDataset, collate_level5_lite
```

### Passo 2: Registrar no Config

Editar `crossattention_split_analysis/config.py`:

```python
# Adicionar ao TrainingConfig
@dataclass
class TrainingConfig:
    # ... existing fields ...
    model_variant: Literal[
        'cnn_crossattn',
        'cross_attention_lite',
        'diffusion',
        'level5_lite'  # NOVO
    ] = 'cnn_crossattn'
```

### Passo 3: Integrar no Experiment

Editar `crossattention_split_analysis/experiment.py`:

```python
def create_model(config: TrainingConfig, ...):
    if config.model_variant == 'level5_lite':
        from .models.level5_lite import Level5LiteModel
        return Level5LiteModel(
            protein_input_dim=config.protein_dim,
            ligand_input_dim=config.ligand_dim,
            hidden_dim=config.hidden_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_cross_attn_layers=config.num_cross_attn_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
    # ... existing model creation ...
```

### Passo 4: Adicionar Dataset Loader

Editar para usar `Level5LiteDataset` quando `model_variant == 'level5_lite'`.

### Passo 5: Integrar no CLI Principal

Ver seção 6.

---

## 6. Integração com CLI

### 6.1 Modificar `semantic_screening_models_beta.py`

#### Adicionar Level 5 às constantes:

```python
LEVEL_LABELS = {
    "level1_fp_knn": "Level 1 (FP+KNN)",
    "level1_fp_mlp": "Level 1 (FP+MLP)",
    "level2_emb_knn": "Level 2 (Emb+KNN)",
    "level2_emb_mlp": "Level 2 (Emb+MLP)",
    "level3_cnn": "Level 3 (CNN)",
    "level4_cnn_ca": "Level 4 (CNN+CA)",
    "level5_lite": "Level 5 (Lite)",  # NOVO
}

LEVEL_COLORS = {
    # ... existing ...
    "level5_lite": "#ff7f0e",  # Orange
}
```

#### Adicionar step no BenchmarkProgress:

```python
if 5 in levels:
    self.steps.append("Step 5: Level 5-Lite")
```

#### Adicionar função `run_level5`:

```python
def run_level5_lite(
    dataset: str,
    embedding_name: str,
    embedding_short: str,
    output_dir: str,
    scaffold_split_dir: str,
    seeds: List[int],
    force: bool,
    epochs: int,
    batch_size: int,
    patience: Optional[int],
    learning_rate: float,
) -> Optional[Dict]:
    """Run Level 5-Lite: Transformer + Cross-Attention."""
    from crossattention_split_analysis.experiment import run_single_analysis
    
    level_dir = os.path.join(output_dir, f"level5_lite_{embedding_short}")
    print(f"  Output: {level_dir}")
    
    results = run_single_analysis(
        embedding_name=embedding_name,
        dataset_type=dataset,
        output_dir=level_dir,
        seeds=seeds,
        force=force,
        scenarios=["scaffold"],
        num_epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        classification_only=True,
        use_molformer_ligand=True,
        scaffold_split_dir=scaffold_split_dir,
        model_variant="level5_lite",  # NOVO
        # Level 5-Lite specific params
        num_encoder_layers=2,
        num_cross_attn_layers=1,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
    )
    
    if results is None:
        results = _load_crossattention_results(level_dir, dataset, embedding_short)
    
    return results
```

#### Adicionar ao main():

```python
# Step 5: Level 5-Lite
level5_results = None
if 5 in levels:
    step_name = "Step 5: Level 5-Lite"
    progress.begin_step(step_name)
    level5_results = run_level5_lite(
        dataset=dataset,
        embedding_name=embedding_name,
        embedding_short=embedding_short,
        output_dir=output_dir,
        scaffold_split_dir=scaffold_split_dir,
        seeds=seeds,
        force=force,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=patience,
        learning_rate=args.learning_rate,
    )
    if level5_results:
        tqdm.write("  Level 5-Lite completed successfully.")
    else:
        tqdm.write("  WARNING: Level 5-Lite returned no results.")
    progress.end_step(step_name)
```

### 6.2 Uso via CLI

```bash
# Rodar apenas Level 5-Lite
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 5 \
    --epochs 200 \
    --batch_size 32 \
    --patience 15

# Comparar todos os levels
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 1,2,5 \
    --epochs 200

# Com seeds específicas
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 5 \
    --seeds 42 123 456
```

---

## 7. Hiperparâmetros

### 7.1 Configuração Default

```python
Level5LiteConfig = {
    # Arquitetura
    'hidden_dim': 512,
    'num_encoder_layers': 2,
    'num_cross_attn_layers': 1,
    'num_heads': 8,
    'dropout': 0.1,
    'classifier_dropout': 0.3,
    
    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'max_epochs': 200,
    'patience': 15,
    'max_grad_norm': 1.0,
    
    # Data
    'max_protein_len': 1024,
    'max_ligand_len': 256,
    
    # Loss
    'use_focal_loss': False,  # Ativar se class imbalance
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
}
```

### 7.2 Justificativa dos Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `hidden_dim=512` | Balanceia expressividade vs overfitting |
| `num_encoder_layers=2` | ESM-2 já é pré-treinado, pouco refinamento necessário |
| `num_cross_attn_layers=1` | 1 layer é suficiente (papers mostram pouco ganho com mais) |
| `num_heads=8` | Standard para dim=512 (64 per head) |
| `dropout=0.1` | Moderado para encoders |
| `classifier_dropout=0.3` | Mais alto no head (risco de overfitting) |
| `batch_size=32` | Maior possível que cabe em GPU |
| `learning_rate=1e-4` | Standard para fine-tuning de transformers |
| `patience=15` | Permite estabilização mas não desperdiça tempo |

---

## 8. Métricas e Avaliação

### 8.1 Métrica Primária: MCC

**Matthews Correlation Coefficient** é a métrica primária porque:
- Robusto a class imbalance
- Considera todos os 4 quadrantes da confusion matrix
- Range: [-1, +1], onde 0 = random

### 8.2 Métricas Secundárias

| Métrica | Uso |
|---------|-----|
| **AUC-ROC** | Avalia ranking geral |
| **F1-Score** | Balanceia precision/recall |
| **Precision** | Importante para drug discovery (reduzir falsos positivos) |
| **Recall** | Importante para não perder hits |
| **Accuracy** | Referência geral |

### 8.3 Threshold Optimization

```python
# Otimizar threshold no validation set
# Não usar threshold fixo de 0.5

thresholds = np.linspace(0.1, 0.9, 81)
best_mcc = -1
best_threshold = 0.5

for t in thresholds:
    preds = (probs >= t).astype(int)
    mcc = matthews_corrcoef(y_true, preds)
    if mcc > best_mcc:
        best_mcc = mcc
        best_threshold = t
```

### 8.4 Expectativa de Performance

| Métrica | Level 1 (atual) | Level 5-Lite (esperado) |
|---------|-----------------|-------------------------|
| **MCC** | 0.428 | 0.48-0.54 |
| **AUC** | 0.792 | 0.82-0.86 |
| **F1** | 0.630 | 0.66-0.72 |

**Se MCC < 0.45:** Investigar por que cross-attention não ajuda.
**Se MCC > 0.50:** Sucesso, considerar upgrade para GNN.

---

## 9. Checklist de Implementação

### Fase 1: Setup (Dia 1)

- [ ] Criar estrutura de diretórios `level5_lite/`
- [ ] Implementar `encoders.py` (ProteinEncoder, LigandEncoder)
- [ ] Implementar `attention.py` (BidirectionalCrossAttention, AttentionPooling)
- [ ] Implementar `classifier.py` (ClassifierHead)
- [ ] Implementar `model.py` (Level5LiteModel)
- [ ] Escrever testes unitários básicos

### Fase 2: Dataset (Dia 2)

- [ ] Implementar `dataset.py` (Level5LiteDataset, collate)
- [ ] Verificar carregamento de matrizes
- [ ] Testar padding e masking
- [ ] Validar shapes em batch

### Fase 3: Training Loop (Dia 3)

- [ ] Integrar com `experiment.py`
- [ ] Adicionar `model_variant='level5_lite'` ao config
- [ ] Testar forward pass com batch real
- [ ] Verificar gradientes e convergência

### Fase 4: CLI Integration (Dia 4)

- [ ] Adicionar `--levels 5` ao CLI
- [ ] Implementar `run_level5_lite()` no orchestrator
- [ ] Testar end-to-end com 1 seed
- [ ] Verificar salvamento de checkpoints e resultados

### Fase 5: Validação (Dia 5)

- [ ] Rodar 1 seed completa (human, 8M)
- [ ] Comparar MCC com Level 1
- [ ] Analisar attention maps (debug)
- [ ] Ajustar hiperparâmetros se necessário

### Fase 6: Produção (Dias 6-7)

- [ ] Rodar 5 seeds completas
- [ ] Gerar relatório comparativo
- [ ] Atualizar visualizações
- [ ] Documentar resultados

---

## 10. Referências

1. **MolTrans** (Huang et al., Bioinformatics 2021)
   - Cross-attention para drug-target interaction
   - DOI: 10.1093/bioinformatics/btaa880

2. **TargetFormer** (Zhang et al., Nature Communications 2023)
   - GNN + Transformer + Cross-Attention
   - DOI: 10.1038/s41467-023-36765-8

3. **Set Transformer** (Lee et al., ICML 2019)
   - Attention pooling com learnable query
   - arXiv: 1810.00825

4. **ESM-2** (Lin et al., Science 2023)
   - Language model para proteínas
   - DOI: 10.1126/science.ade2574

5. **MoLFormer** (Ross et al., Nature Machine Intelligence 2022)
   - Transformer para moléculas
   - DOI: 10.1038/s42256-022-00580-7

---

*Documento criado em: 2026-03-02*
*Autores: Semantic Screening Team*
*Status: **Pronto para implementação***
