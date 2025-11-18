# Plano de Refatoração: Arquitetura SOLID para Suporte a ESM-3

**Status:** 📋 PLANEJAMENTO  
**Data Criação:** 2025-11-18  
**Branch Relacionado:** `offloader`  
**Pull Request:** #78  
**Autor:** Equipe DocKTKinase  
**Prioridade:** ALTA (Arquitetura para Escalabilidade Futura)

---

## 📌 CONTEXTO E MOTIVAÇÃO

### Situação Atual

O módulo `ProteinEmbedding` está **fortemente acoplado** à implementação específica do **ESM-2**, dificultando:
- ✗ Adição de novos modelos (ex: ESM-3) sem modificar código existente
- ✗ Testes unitários isolados por estratégia de modelo
- ✗ Manutenção independente de implementações ESM-2 vs ESM-3
- ✗ Reutilização de componentes entre diferentes modelos

### Problema Técnico Identificado

```python
# ATUAL: Acoplamento direto no ProteinEmbedding
class ProteinEmbedding(BaseEmbedding):
    def _load_model(self):
        # Lógica específica ESM-2 hardcoded
        model, alphabet = esm.pretrained.load_model_and_alphabet(...)
        
    def _generate_single_embedding(self, sequence):
        # Processamento específico ESM-2
        batch_tokens = alphabet.get_batch_converter(...)
        results = model(tokens, repr_layers=[33])
        embedding = results["representations"][33].mean(1)
```

**Problema:** Adicionar ESM-3 exigiria:
1. Modificar `_load_model()` com if/else para detectar modelo
2. Modificar `_generate_single_embedding()` com lógica condicional
3. Risco de quebrar código ESM-2 ao adicionar ESM-3
4. Violação do princípio Open/Closed (SOLID)

### Objetivo da Refatoração

Implementar **Strategy Pattern + Factory Pattern** para:
- ✓ Desacoplar implementações de modelos específicos
- ✓ Permitir adição de ESM-3 **sem modificar** código ESM-2
- ✓ Facilitar testes unitários isolados por estratégia
- ✓ Manter retrocompatibilidade com pipelines existentes
- ✓ Seguir princípios SOLID (Single Responsibility, Open/Closed)

---

## 🏗️ ARQUITETURA PROPOSTA

### Diagrama de Classes

```
┌─────────────────────────────────────────────────────────────┐
│                      BaseEmbedding                          │
│  (Abstract Base Class - Já Existente)                       │
│  + generate_embedding(sequence)                             │
│  + generate_batch_embeddings(sequences)                     │
│  + process_file(input_file, output_file)                    │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                    ProteinEmbedding                         │
│  (Orchestrator - Refatorado)                                │
│  - strategy: BaseProteinStrategy                            │
│  + __init__(model_name, device)                             │
│  + _load_model() → delega para strategy.load()              │
│  + _generate_single_embedding() → strategy.generate()       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ usa
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              BaseProteinStrategy                            │
│  (Interface - Nova)                                         │
│  + load(model_name, device, **kwargs) → Tuple[model, ...]  │
│  + generate(model, sequence, **kwargs) → np.ndarray         │
│  + get_max_length() → int                                   │
│  + get_embedding_dim() → int                                │
└─────────────────────────────────────────────────────────────┘
                              ▲
                ┌─────────────┴─────────────┐
                │                           │
┌───────────────────────────┐   ┌───────────────────────────┐
│     ESM2Strategy          │   │     ESM3Strategy          │
│  (Implementação Atual)    │   │  (Implementação Futura)   │
│  + load() → ESM-2 model   │   │  + load() → ESM-3 model   │
│  + generate() → embeddings│   │  + generate() → embeddings│
│  + max_length = 5120      │   │  + max_length = ????      │
└───────────────────────────┘   └───────────────────────────┘
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │ ProteinModelFactory│
                    │  (Factory - Novo)  │
                    │ + create_strategy()│
                    └────────────────────┘
```

### Hierarquia de Arquivos Proposta

```
src/build/embeddings/
├── base_embedding.py              # Já existe - não alterar
├── protein_embedding.py           # REFATORAR (orchestrator)
├── strategies/
│   ├── __init__.py
│   ├── base_protein_strategy.py   # CRIAR (interface ABC)
│   ├── esm2_strategy.py           # CRIAR (extrair código atual)
│   └── esm3_strategy.py           # CRIAR (futuro, placeholder)
└── factories/
    ├── __init__.py
    └── protein_model_factory.py   # CRIAR (detectar e criar strategy)
```

---

## 📝 ESPECIFICAÇÃO DETALHADA DOS COMPONENTES

### 1. BaseProteinStrategy (Interface ABC)

**Localização:** `src/build/embeddings/strategies/base_protein_strategy.py`

```python
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import numpy as np
import torch

class BaseProteinStrategy(ABC):
    """
    Interface abstrata para estratégias de modelos de proteína.
    Cada modelo (ESM-2, ESM-3, etc.) implementa esta interface.
    """
    
    @abstractmethod
    def load(
        self, 
        model_name: str, 
        device: torch.device,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Carrega modelo e componentes necessários (alphabet, tokenizer, etc.)
        
        Args:
            model_name: Nome do modelo (ex: "esm2_t48_15B_UR50D")
            device: Dispositivo PyTorch (cuda/cpu)
            offload_folder: Pasta para CPU offloading (opcional)
            **kwargs: Parâmetros específicos do modelo
            
        Returns:
            Tuple contendo (model, auxiliary_objects)
            - auxiliary_objects pode ser alphabet, tokenizer, etc.
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Gera embedding para uma sequência de proteína.
        
        Args:
            model: Modelo carregado
            auxiliary_objects: Objetos auxiliares (alphabet, tokenizer)
            sequence: Sequência de aminoácidos
            device: Dispositivo PyTorch
            **kwargs: Parâmetros específicos (layers, pooling, etc.)
            
        Returns:
            Embedding numpy array (shape: [embedding_dim])
        """
        pass
    
    @abstractmethod
    def get_max_length(self, model_name: str) -> int:
        """Retorna comprimento máximo de sequência para o modelo."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self, model_name: str) -> int:
        """Retorna dimensão do embedding gerado."""
        pass
    
    @abstractmethod
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """
        Libera recursos (memória GPU, tensors, etc.)
        Chamado após generate() para garantir limpeza.
        """
        pass
```

---

### 2. ESM2Strategy (Implementação Concreta)

**Localização:** `src/build/embeddings/strategies/esm2_strategy.py`

```python
import torch
import numpy as np
import gc
import esm
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from typing import Tuple, Any, Optional
from .base_protein_strategy import BaseProteinStrategy
from ...core.constants import ESM_MODELS

class ESM2Strategy(BaseProteinStrategy):
    """
    Estratégia de implementação para modelos ESM-2.
    Contém toda a lógica específica de ESM-2 (offloading, alphabet, etc.)
    """
    
    def load(
        self, 
        model_name: str, 
        device: torch.device,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Carrega modelo ESM-2 com CPU offloading."""
        
        # Verificação de modelo suportado
        if model_name not in ESM_MODELS:
            raise ValueError(f"Modelo ESM-2 '{model_name}' não encontrado.")
        
        # Carregar modelo e alphabet
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.eval()
        
        # CPU Offloading para modelos grandes (15B)
        if "15B" in model_name:
            max_memory = get_balanced_memory(
                model,
                max_memory={0: "20GiB", "cpu": "30GiB"},
                no_split_module_classes=["TransformerLayer"]
            )
            
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["TransformerLayer"]
            )
            
            model = dispatch_model(
                model,
                device_map=device_map,
                offload_folder=offload_folder
            )
        else:
            model = model.to(device)
        
        return model, alphabet
    
    def generate(
        self,
        model: Any,
        auxiliary_objects: Any,  # alphabet neste caso
        sequence: str,
        device: torch.device,
        repr_layer: int = 33,
        pooling: str = "mean",
        **kwargs
    ) -> np.ndarray:
        """Gera embedding ESM-2 com mean pooling."""
        
        alphabet = auxiliary_objects
        max_len = self.get_max_length(model.args.arch)
        
        # Truncar sequência se necessário
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        
        # Converter para tokens
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
        batch_tokens = batch_tokens.to(device)
        
        # Inferência
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])
            token_representations = results["representations"][repr_layer]
            
            # Mean pooling (ignorar tokens especiais <cls> e <eos>)
            if pooling == "mean":
                embedding = token_representations[0, 1:len(sequence)+1].mean(0)
            else:
                raise ValueError(f"Pooling '{pooling}' não suportado.")
            
            embedding_np = embedding.cpu().numpy()
        
        # Limpeza de memória
        del batch_tokens, results, token_representations, embedding
        torch.cuda.empty_cache()
        gc.collect()
        
        return embedding_np
    
    def get_max_length(self, model_name: str) -> int:
        """Retorna max_len do ESM_MODELS."""
        return ESM_MODELS.get(model_name, {}).get("max_len", 1024)
    
    def get_embedding_dim(self, model_name: str) -> int:
        """Retorna dimensão do embedding."""
        return ESM_MODELS.get(model_name, {}).get("dim", 1280)
    
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """Libera memória após processamento."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
```

---

### 3. ProteinModelFactory

**Localização:** `src/build/embeddings/factories/protein_model_factory.py`

```python
import torch
from typing import Optional
from ..strategies.base_protein_strategy import BaseProteinStrategy
from ..strategies.esm2_strategy import ESM2Strategy
# from ..strategies.esm3_strategy import ESM3Strategy  # Futuro

class ProteinModelFactory:
    """
    Factory para criar estratégias de modelos de proteína.
    Detecta qual strategy usar baseado no nome do modelo.
    """
    
    @staticmethod
    def create_strategy(model_name: str) -> BaseProteinStrategy:
        """
        Cria strategy apropriada baseada no nome do modelo.
        
        Args:
            model_name: Nome do modelo (ex: "esm2_t48_15B_UR50D")
            
        Returns:
            Instância de BaseProteinStrategy (ESM2Strategy, ESM3Strategy, etc.)
            
        Raises:
            ValueError: Se modelo não for reconhecido
        """
        
        # Detectar ESM-2
        if model_name.startswith("esm2") or model_name.startswith("esm1"):
            return ESM2Strategy()
        
        # Detectar ESM-3 (futuro)
        # if model_name.startswith("esm3"):
        #     return ESM3Strategy()
        
        raise ValueError(
            f"Modelo '{model_name}' não reconhecido. "
            f"Estratégias suportadas: ESM-2 (esm2_*, esm1b_*)"
        )
```

---

### 4. ProteinEmbedding Refatorado

**Localização:** `src/build/embeddings/protein_embedding.py`

```python
class ProteinEmbedding(BaseEmbedding):
    """
    Orchestrator para embeddings de proteínas.
    Delega operações para strategy específica (ESM2, ESM3, etc.)
    """
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = "cuda"):
        super().__init__(device)
        self.model_name = model_name
        
        # Factory cria strategy apropriada
        from .factories.protein_model_factory import ProteinModelFactory
        self.strategy = ProteinModelFactory.create_strategy(model_name)
        
        self.model = None
        self.auxiliary_objects = None  # alphabet, tokenizer, etc.
        self._load_model()
    
    def _load_model(self) -> None:
        """Carrega modelo usando strategy."""
        offload_folder = "./tmp/offload" if "15B" in self.model_name else None
        
        self.model, self.auxiliary_objects = self.strategy.load(
            model_name=self.model_name,
            device=self.device,
            offload_folder=offload_folder
        )
    
    def _generate_single_embedding(self, sequence: str) -> np.ndarray:
        """Gera embedding usando strategy."""
        return self.strategy.generate(
            model=self.model,
            auxiliary_objects=self.auxiliary_objects,
            sequence=sequence,
            device=self.device
        )
    
    def __del__(self):
        """Cleanup ao destruir objeto."""
        if self.strategy and self.model:
            self.strategy.cleanup(self.model, self.auxiliary_objects)
```

---

## 🧪 PLANO DE TESTES EXAUSTIVO

### Estrutura de Testes

```
tests/
├── unit/
│   ├── test_esm2_strategy.py           # Testes isolados ESM-2
│   ├── test_protein_model_factory.py   # Testes factory
│   └── test_protein_embedding.py       # Testes integração
├── integration/
│   ├── test_pipeline_esm2.py           # Pipeline completo ESM-2
│   └── test_memory_management.py       # Testes memória/offloading
└── regression/
    └── test_backward_compatibility.py  # Garantir retrocompat.
```

---

### 1. Testes Unitários: ESM2Strategy

**Arquivo:** `tests/unit/test_esm2_strategy.py`

```python
import pytest
import torch
import numpy as np
from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy

class TestESM2Strategy:
    """Testes isolados para ESM2Strategy."""
    
    @pytest.fixture
    def strategy(self):
        return ESM2Strategy()
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===== TESTES DE CARREGAMENTO =====
    
    def test_load_small_model_success(self, strategy, device):
        """Teste: Carregar modelo pequeno (8M) sem offloading."""
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        
        assert model is not None
        assert alphabet is not None
        assert hasattr(model, 'eval')
    
    def test_load_15B_with_offloading(self, strategy, device):
        """Teste: Carregar 15B com CPU offloading."""
        if not torch.cuda.is_available():
            pytest.skip("Requer GPU para testar offloading")
        
        model, alphabet = strategy.load(
            "esm2_t48_15B_UR50D", 
            device, 
            offload_folder="./tmp/test_offload"
        )
        
        assert model is not None
        # Verificar se offloading foi aplicado
        assert hasattr(model, 'hf_device_map')
    
    def test_load_invalid_model_raises_error(self, strategy, device):
        """Teste: Modelo inválido lança ValueError."""
        with pytest.raises(ValueError, match="não encontrado"):
            strategy.load("modelo_inexistente", device)
    
    # ===== TESTES DE GERAÇÃO =====
    
    def test_generate_embedding_shape(self, strategy, device):
        """Teste: Embedding tem shape correto."""
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
        
        embedding = strategy.generate(model, alphabet, sequence, device)
        
        expected_dim = strategy.get_embedding_dim("esm2_t6_8M_UR50D")
        assert embedding.shape == (expected_dim,)
        assert isinstance(embedding, np.ndarray)
    
    def test_generate_with_truncation(self, strategy, device):
        """Teste: Sequências longas são truncadas."""
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        long_sequence = "A" * 10000  # Muito longa
        
        embedding = strategy.generate(model, alphabet, long_sequence, device)
        
        # Deve gerar embedding sem erro
        assert embedding is not None
        assert len(embedding.shape) == 1
    
    def test_generate_empty_sequence_raises_error(self, strategy, device):
        """Teste: Sequência vazia lança erro."""
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        
        with pytest.raises(Exception):
            strategy.generate(model, alphabet, "", device)
    
    def test_generate_invalid_amino_acids(self, strategy, device):
        """Teste: Aminoácidos inválidos são tratados."""
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        invalid_sequence = "MKTXYZ123"  # X, Y, Z são válidos; 1,2,3 não
        
        # ESM-2 converte caracteres inválidos para <unk>
        embedding = strategy.generate(model, alphabet, invalid_sequence, device)
        assert embedding is not None
    
    # ===== TESTES DE CONFIGURAÇÃO =====
    
    @pytest.mark.parametrize("model_name,expected_max_len", [
        ("esm2_t6_8M_UR50D", 1024),
        ("esm2_t33_650M_UR50D", 1024),
        ("esm2_t36_3B_UR50D", 4096),
        ("esm2_t48_15B_UR50D", 5120),
    ])
    def test_get_max_length(self, strategy, model_name, expected_max_len):
        """Teste: get_max_length retorna valores corretos."""
        assert strategy.get_max_length(model_name) == expected_max_len
    
    @pytest.mark.parametrize("model_name,expected_dim", [
        ("esm2_t6_8M_UR50D", 320),
        ("esm2_t33_650M_UR50D", 1280),
        ("esm2_t36_3B_UR50D", 2560),
        ("esm2_t48_15B_UR50D", 5120),
    ])
    def test_get_embedding_dim(self, strategy, model_name, expected_dim):
        """Teste: get_embedding_dim retorna valores corretos."""
        assert strategy.get_embedding_dim(model_name) == expected_dim
    
    # ===== TESTES DE MEMÓRIA =====
    
    def test_cleanup_frees_memory(self, strategy, device):
        """Teste: cleanup() libera memória."""
        if not torch.cuda.is_available():
            pytest.skip("Requer GPU para testar limpeza CUDA")
        
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        
        # Capturar memória antes
        initial_memory = torch.cuda.memory_allocated()
        
        # Gerar embedding
        sequence = "MKTAYIAKQRQISFVK"
        strategy.generate(model, alphabet, sequence, device)
        
        # Cleanup
        strategy.cleanup(model, alphabet)
        torch.cuda.synchronize()
        
        # Verificar que memória foi liberada
        final_memory = torch.cuda.memory_allocated()
        # Memória final deve ser <= inicial (permite pequenas variações)
        assert final_memory <= initial_memory * 1.1
    
    def test_generate_multiple_sequences_no_oom(self, strategy, device):
        """Teste: Gerar múltiplos embeddings sem OOM."""
        if not torch.cuda.is_available():
            pytest.skip("Requer GPU")
        
        model, alphabet = strategy.load("esm2_t6_8M_UR50D", device)
        sequences = ["MKTAYIAK" * 50] * 100  # 100 sequências médias
        
        for seq in sequences:
            embedding = strategy.generate(model, alphabet, seq, device)
            assert embedding is not None
        
        # Sem OOM = sucesso
```

---

### 2. Testes Unitários: ProteinModelFactory

**Arquivo:** `tests/unit/test_protein_model_factory.py`

```python
import pytest
from src.build.embeddings.factories.protein_model_factory import ProteinModelFactory
from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy

class TestProteinModelFactory:
    """Testes para ProteinModelFactory."""
    
    # ===== TESTES DE CRIAÇÃO ESM-2 =====
    
    @pytest.mark.parametrize("model_name", [
        "esm2_t6_8M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D",
        "esm1b_t33_650M_UR50S",
    ])
    def test_create_esm2_strategy(self, model_name):
        """Teste: Factory cria ESM2Strategy para modelos ESM-2."""
        strategy = ProteinModelFactory.create_strategy(model_name)
        assert isinstance(strategy, ESM2Strategy)
    
    # ===== TESTES DE ERRO =====
    
    def test_create_unknown_model_raises_error(self):
        """Teste: Modelo desconhecido lança ValueError."""
        with pytest.raises(ValueError, match="não reconhecido"):
            ProteinModelFactory.create_strategy("gpt4_protein_model")
    
    def test_create_empty_name_raises_error(self):
        """Teste: Nome vazio lança erro."""
        with pytest.raises(ValueError):
            ProteinModelFactory.create_strategy("")
    
    # ===== TESTES DE EXTENSIBILIDADE (Futuro ESM-3) =====
    
    @pytest.mark.skip("ESM-3 ainda não implementado")
    def test_create_esm3_strategy(self):
        """Teste: Factory cria ESM3Strategy (quando implementado)."""
        strategy = ProteinModelFactory.create_strategy("esm3_base")
        # assert isinstance(strategy, ESM3Strategy)
```

---

### 3. Testes Unitários: ProteinEmbedding Refatorado

**Arquivo:** `tests/unit/test_protein_embedding.py`

```python
import pytest
import torch
import numpy as np
from src.build.embeddings.protein_embedding import ProteinEmbedding

class TestProteinEmbedding:
    """Testes para ProteinEmbedding refatorado."""
    
    @pytest.fixture
    def embedding_generator(self):
        return ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
    
    # ===== TESTES DE INTEGRAÇÃO COM STRATEGY =====
    
    def test_init_creates_strategy(self, embedding_generator):
        """Teste: __init__ cria strategy correta."""
        assert embedding_generator.strategy is not None
        assert embedding_generator.model is not None
        assert embedding_generator.auxiliary_objects is not None
    
    def test_generate_single_embedding(self, embedding_generator):
        """Teste: Gerar embedding único."""
        sequence = "MKTAYIAKQRQISFVK"
        embedding = embedding_generator._generate_single_embedding(sequence)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (320,)  # 8M model dim
    
    def test_generate_batch_embeddings(self, embedding_generator):
        """Teste: Gerar batch de embeddings."""
        sequences = [
            "MKTAYIAKQRQISFVK",
            "ARHPHILNEQVAEVAEALR",
            "GLSDGEWQQVLNVWGKVE"
        ]
        
        embeddings = embedding_generator.generate_batch_embeddings(sequences)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (320,) for emb in embeddings)
    
    # ===== TESTES DE COMPATIBILIDADE =====
    
    def test_backward_compatibility_api(self, embedding_generator):
        """Teste: API pública mantida (retrocompatibilidade)."""
        # Verificar que métodos públicos existem
        assert hasattr(embedding_generator, 'generate_embedding')
        assert hasattr(embedding_generator, 'generate_batch_embeddings')
        assert hasattr(embedding_generator, 'process_file')
    
    # ===== TESTES DE DIFERENTES MODELOS =====
    
    @pytest.mark.parametrize("model_name", [
        "esm2_t6_8M_UR50D",
        "esm2_t33_650M_UR50D",
    ])
    def test_multiple_models(self, model_name):
        """Teste: Carregar diferentes modelos ESM-2."""
        gen = ProteinEmbedding(model_name=model_name)
        sequence = "MKTAYIAKQRQISFVK"
        
        embedding = gen._generate_single_embedding(sequence)
        assert embedding is not None
```

---

### 4. Testes de Integração: Pipeline Completo

**Arquivo:** `tests/integration/test_pipeline_esm2.py`

```python
import pytest
import pandas as pd
import os
from src.build.embeddings.protein_embedding import ProteinEmbedding

class TestPipelineESM2:
    """Testes de pipeline completo com ESM-2."""
    
    @pytest.fixture
    def test_data_file(self, tmp_path):
        """Cria arquivo TSV de teste."""
        data = {
            'uniprot_id': ['P12345', 'Q67890', 'R11111'],
            'sequence': [
                'MKTAYIAKQRQISFVK',
                'ARHPHILNEQVAEVAEALR',
                'GLSDGEWQQVLNVWGKVE'
            ]
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "test_proteins.tsv"
        df.to_csv(file_path, sep='\t', index=False)
        return str(file_path)
    
    def test_process_file_8M_model(self, test_data_file, tmp_path):
        """Teste: Processar arquivo com modelo 8M."""
        output_file = tmp_path / "output_embeddings.npz"
        
        generator = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
        generator.process_file(test_data_file, str(output_file))
        
        # Verificar output
        assert os.path.exists(output_file)
        
        data = np.load(output_file)
        assert 'P12345' in data
        assert 'Q67890' in data
        assert 'R11111' in data
        assert data['P12345'].shape == (320,)
    
    @pytest.mark.slow
    def test_process_large_dataset(self, tmp_path):
        """Teste: Processar dataset com 100 sequências."""
        # Criar dataset grande
        sequences = ["MKTAYIAK" * 50] * 100
        uniprot_ids = [f"P{i:05d}" for i in range(100)]
        
        df = pd.DataFrame({'uniprot_id': uniprot_ids, 'sequence': sequences})
        input_file = tmp_path / "large_dataset.tsv"
        df.to_csv(input_file, sep='\t', index=False)
        
        output_file = tmp_path / "large_output.npz"
        
        generator = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
        generator.process_file(str(input_file), str(output_file))
        
        # Verificar que todos foram processados
        data = np.load(output_file)
        assert len(data.files) == 100
    
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_process_file_15B_with_offloading(self, test_data_file, tmp_path):
        """Teste: Processar com modelo 15B e CPU offloading."""
        if not torch.cuda.is_available():
            pytest.skip("Requer GPU")
        
        output_file = tmp_path / "output_15B.npz"
        
        generator = ProteinEmbedding(model_name="esm2_t48_15B_UR50D")
        generator.process_file(test_data_file, str(output_file))
        
        # Verificar dimensão correta (5120)
        data = np.load(output_file)
        assert data['P12345'].shape == (5120,)
```

---

### 5. Testes de Regressão: Retrocompatibilidade

**Arquivo:** `tests/regression/test_backward_compatibility.py`

```python
import pytest
import numpy as np
from src.build.embeddings.protein_embedding import ProteinEmbedding

class TestBackwardCompatibility:
    """Garante que refatoração não quebra código existente."""
    
    def test_old_api_still_works(self):
        """Teste: API antiga continua funcional."""
        # Código que usuários existentes podem ter
        generator = ProteinEmbedding(model_name="esm2_t6_8M_UR50D", device="cuda")
        sequence = "MKTAYIAKQRQISFVK"
        
        # Método público ainda existe e funciona
        embedding = generator.generate_embedding(sequence)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
    
    def test_embeddings_match_original(self):
        """Teste: Embeddings gerados são idênticos à versão original."""
        # Carregar embedding de referência (gerado com código antigo)
        # reference_embedding = np.load("tests/fixtures/reference_embedding.npy")
        
        generator = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
        sequence = "MKTAYIAKQRQISFVK"
        
        new_embedding = generator.generate_embedding(sequence)
        
        # Embeddings devem ser numericamente idênticos
        # np.testing.assert_allclose(new_embedding, reference_embedding, rtol=1e-5)
        
        # Por enquanto, apenas verificar que gerou corretamente
        assert new_embedding.shape == (320,)
    
    def test_process_file_output_format_unchanged(self, tmp_path):
        """Teste: Formato do arquivo de saída não mudou."""
        # Simular dataset
        import pandas as pd
        data = {'uniprot_id': ['P12345'], 'sequence': ['MKTAYIAK']}
        df = pd.DataFrame(data)
        
        input_file = tmp_path / "test.tsv"
        output_file = tmp_path / "output.npz"
        df.to_csv(input_file, sep='\t', index=False)
        
        generator = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
        generator.process_file(str(input_file), str(output_file))
        
        # Formato deve ser npz com keys = uniprot_ids
        data = np.load(output_file)
        assert 'P12345' in data
        assert isinstance(data['P12345'], np.ndarray)
```

---

### 6. Testes de Memória e Performance

**Arquivo:** `tests/integration/test_memory_management.py`

```python
import pytest
import torch
from src.build.embeddings.protein_embedding import ProteinEmbedding

class TestMemoryManagement:
    """Testes de gerenciamento de memória."""
    
    @pytest.mark.gpu
    def test_no_memory_leak_after_100_sequences(self):
        """Teste: Sem vazamento de memória após 100 sequências."""
        if not torch.cuda.is_available():
            pytest.skip("Requer GPU")
        
        generator = ProteinEmbedding(model_name="esm2_t6_8M_UR50D")
        sequences = ["MKTAYIAK" * 50] * 100
        
        initial_memory = torch.cuda.memory_allocated()
        
        for seq in sequences:
            generator.generate_embedding(seq)
        
        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()
        
        # Memória não deve crescer mais de 10%
        memory_increase = final_memory - initial_memory
        assert memory_increase < initial_memory * 0.1
    
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_15B_processes_299_sequences_without_oom(self):
        """Teste: Modelo 15B processa 299 sequências (caso real)."""
        if not torch.cuda.is_available():
            pytest.skip("Requer GPU")
        
        generator = ProteinEmbedding(model_name="esm2_t48_15B_UR50D")
        
        # Simular 299 sequências (tamanho real do dataset)
        sequences = ["MKTAYIAK" * 100] * 299
        
        for i, seq in enumerate(sequences):
            embedding = generator.generate_embedding(seq)
            assert embedding is not None
            
            if (i + 1) % 50 == 0:
                print(f"Processadas {i+1}/299 sequências")
        
        # Se chegou aqui sem OOM = sucesso
```

---

## 🔄 FASES DE IMPLEMENTAÇÃO

### Fase 1: Criar Abstrações (1-2 dias)
- [ ] Criar `base_protein_strategy.py` (interface ABC)
- [ ] Criar estrutura de pastas `strategies/` e `factories/`
- [ ] Adicionar docstrings completas
- [ ] **Teste:** Verificar que interface ABC não pode ser instanciada

### Fase 2: Implementar ESM2Strategy (2-3 dias)
- [ ] Extrair código de `ProteinEmbedding` para `ESM2Strategy`
- [ ] Implementar todos os métodos abstratos
- [ ] Adicionar tratamento de erros específicos
- [ ] **Testes:** Executar `test_esm2_strategy.py` (21 testes)

### Fase 3: Criar Factory (1 dia)
- [ ] Implementar `ProteinModelFactory.create_strategy()`
- [ ] Adicionar detecção de modelos ESM-2/ESM-3
- [ ] **Testes:** Executar `test_protein_model_factory.py` (5 testes)

### Fase 4: Refatorar ProteinEmbedding (2-3 dias)
- [ ] Substituir lógica direta por delegação à strategy
- [ ] Manter API pública inalterada (retrocompat.)
- [ ] Adicionar cleanup no `__del__`
- [ ] **Testes:** Executar `test_protein_embedding.py` (6 testes)

### Fase 5: Testes de Integração (2-3 dias)
- [ ] Executar pipeline completo com modelo 8M
- [ ] Executar pipeline com modelo 15B + offloading
- [ ] Testar com dataset de 100 sequências
- [ ] **Testes:** Executar `test_pipeline_esm2.py` (3 testes)

### Fase 6: Testes de Regressão (1-2 dias)
- [ ] Gerar embeddings de referência com código antigo
- [ ] Comparar embeddings novos vs. antigos (tolerância 1e-5)
- [ ] Verificar formato de saída `process_file()`
- [ ] **Testes:** Executar `test_backward_compatibility.py` (3 testes)

### Fase 7: Testes de Memória (2-3 dias)
- [ ] Testar 100 sequências sem vazamento
- [ ] Testar 299 sequências com 15B (caso real OOM)
- [ ] Monitorar com `nvidia-smi` durante teste
- [ ] **Testes:** Executar `test_memory_management.py` (2 testes)

### Fase 8: Documentação e Review (1-2 dias)
- [ ] Atualizar README com nova arquitetura
- [ ] Criar guia "Como adicionar ESM-3"
- [ ] Code review completo
- [ ] Merge para `main`

---

## ✅ CRITÉRIOS DE SUCESSO

### Funcionalidade
- ✅ Todos os 40+ testes passando
- ✅ Pipeline processa 299 sequências sem OOM
- ✅ Embeddings idênticos ao código original (rtol=1e-5)

### Qualidade de Código
- ✅ Cobertura de testes ≥ 90%
- ✅ Type hints em todas as funções
- ✅ Docstrings seguindo Google Style
- ✅ Sem warnings do pylint/mypy

### Performance
- ✅ Sem regressão de tempo (±5% do original)
- ✅ Sem vazamento de memória em 1000 sequências
- ✅ CPU offloading funcional para 15B

### Manutenibilidade
- ✅ Adicionar ESM-3 requer apenas criar `ESM3Strategy` (sem modificar ESM-2)
- ✅ Factory detecta automaticamente novo modelo
- ✅ Documentação clara para extensibilidade

---

## 📊 ESTIMATIVA DE ESFORÇO

| Fase                      | Dias  | Complexidade |
|---------------------------|-------|--------------|
| Fase 1: Abstrações        | 1-2   | Baixa        |
| Fase 2: ESM2Strategy      | 2-3   | Alta         |
| Fase 3: Factory           | 1     | Baixa        |
| Fase 4: Refatorar PE      | 2-3   | Média        |
| Fase 5: Testes Integração | 2-3   | Média        |
| Fase 6: Testes Regressão  | 1-2   | Baixa        |
| Fase 7: Testes Memória    | 2-3   | Alta         |
| Fase 8: Documentação      | 1-2   | Baixa        |
| **TOTAL**                 | **12-19 dias** | |

---

## 🚨 RISCOS E MITIGAÇÕES

### Risco 1: Quebrar Código Existente
- **Probabilidade:** Média
- **Impacto:** Alto
- **Mitigação:** Testes de regressão exaustivos + manter API pública inalterada

### Risco 2: Performance Degradation
- **Probabilidade:** Baixa
- **Impacto:** Médio
- **Mitigação:** Benchmarks antes/depois + profiling

### Risco 3: Complexidade Excessiva
- **Probabilidade:** Baixa
- **Impacto:** Médio
- **Mitigação:** Manter design simples + documentação clara

### Risco 4: OOM Persistir Após Refatoração
- **Probabilidade:** Baixa
- **Impacto:** Alto
- **Mitigação:** Manter estratégias de memory cleanup atuais + testes específicos

---

## 📚 REFERÊNCIAS

1. **Design Patterns:**
   - Strategy Pattern: https://refactoring.guru/design-patterns/strategy
   - Factory Pattern: https://refactoring.guru/design-patterns/factory-method
   
2. **SOLID Principles:**
   - Single Responsibility: https://en.wikipedia.org/wiki/Single-responsibility_principle
   - Open/Closed: https://en.wikipedia.org/wiki/Open–closed_principle
   
3. **ESM Models:**
   - ESM-2 Paper: https://www.science.org/doi/10.1126/science.ade2574
   - ESM-3 (futuro): https://www.evolutionaryscale.ai/

4. **PyTorch Best Practices:**
   - Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
   - Accelerate Library: https://huggingface.co/docs/accelerate/

---

## 📝 NOTAS FINAIS

### Quando Implementar?
- **AGORA:** Se ESM-3 release está próximo (3-6 meses)
- **DEPOIS:** Se ESM-3 ainda está distante (>1 ano)

### Alternativa Mais Simples (Se Não Urgente)
Se o foco imediato é apenas resolver OOM (não adicionar ESM-3 agora):
1. Manter arquitetura atual
2. Focar em otimizar memory cleanup
3. Adiar refatoração para quando ESM-3 for realmente necessário

### Discussão Necessária
- **Escopo:** Refatorar agora ou aguardar ESM-3?
- **Retrocompat.:** Manter 100% da API antiga?
- **Migration:** Criar novo módulo ou refatorar in-place?

---

**Status Final:** 📋 DOCUMENTO SALVO - Aguardando Decisão  
**Próximo Passo:** Discutir com equipe → Aprovar escopo → Iniciar Fase 1
