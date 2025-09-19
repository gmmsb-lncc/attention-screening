# ESTRATÉGIA DE MODULARIZAÇÃO - PASTA BUILD

## 📊 ARQUITETURA PROPOSTA

### 🎯 Objetivos da Modularização:
1. **Separação clara de responsabilidades**
2. **Reutilização de componentes**
3. **Facilidade de manutenção e testes**
4. **Extensibilidade para novos tipos de embeddings**
5. **Melhor gestão de dependências**

### 🏗️ Nova Estrutura Modular:

```
src/build/
├── __init__.py                    # Módulo principal
├── core/                          # Funcionalidades principais
│   ├── __init__.py
│   ├── base_builder.py            # Classe base abstrata
│   ├── config.py                  # Configurações globais
│   ├── constants.py               # Constantes do sistema
│   └── exceptions.py              # Exceções customizadas
├── embeddings/                    # Geração de embeddings
│   ├── __init__.py
│   ├── base_embedding.py          # Interface comum
│   ├── protein_embedding.py       # Embeddings de proteínas (Meta/ESM)
│   ├── ligand_embedding.py        # Embeddings de ligantes (IBM/FM4M)
│   └── utils.py                   # Utilitários para embeddings
├── matrix/                        # Construção de matrizes
│   ├── __init__.py
│   ├── base_matrix.py             # Interface de matriz
│   ├── embedding_matrix.py        # Matriz de embeddings
│   ├── kinase_matrix.py           # Matriz de kinases
│   └── matrix_utils.py            # Utilitários de matriz
├── labels/                        # Geração de labels
│   ├── __init__.py
│   ├── base_labels.py             # Interface de labels
│   ├── interaction_labels.py      # Labels de interação
│   ├── binary_labels.py           # Labels binários
│   └── label_utils.py             # Utilitários de labels
├── validation/                    # Validação e verificação
│   ├── __init__.py
│   ├── base_validator.py          # Interface de validação
│   ├── embedding_validator.py     # Validação de embeddings
│   ├── matrix_validator.py        # Validação de matrizes
│   └── validation_utils.py        # Utilitários de validação
├── pipeline/                      # Orquestração e pipelines
│   ├── __init__.py
│   ├── build_pipeline.py          # Pipeline principal
│   ├── embedding_pipeline.py      # Pipeline de embeddings
│   └── pipeline_utils.py          # Utilitários de pipeline
└── utils/                         # Utilitários gerais
    ├── __init__.py
    ├── file_utils.py              # Manipulação de arquivos
    ├── memory_utils.py            # Gestão de memória
    ├── spark_utils.py             # Utilitários Spark
    └── logging_utils.py           # Sistema de logging
```

### 🔄 Mapeamento de Arquivos Atuais:

**MIGRAÇÃO DE FUNCIONALIDADES:**

1. **embeddingMeta.py** → `embeddings/protein_embedding.py`
2. **embeddingIBM.py** → `embeddings/ligand_embedding.py`
3. **buildEmbeddingMatrix.py** → `matrix/embedding_matrix.py`
4. **buildKinaseMatrix.py** → `matrix/kinase_matrix.py`
5. **buildInteractionLabels.py** → `labels/interaction_labels.py`
6. **buildbinaryLabels.py** → `labels/binary_labels.py`
7. **checkEmbedding.py** → `validation/embedding_validator.py`
8. **checkConcatenate.py** → `validation/matrix_validator.py`
9. **build.py** → `pipeline/build_pipeline.py`
10. **embeddingBuild.py** → `pipeline/embedding_pipeline.py`

### 🧩 Interfaces e Abstrações:

#### 1. **BaseBuilder** (core/base_builder.py):
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseBuilder(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        pass
    
    @abstractmethod
    def build(self) -> Any:
        pass
    
    @abstractmethod
    def save(self, output_path: str) -> None:
        pass
```

#### 2. **BaseEmbedding** (embeddings/base_embedding.py):
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np

class BaseEmbedding(ABC):
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model = None
    
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def generate_embedding(self, input_data: Union[str, List[str]]) -> np.ndarray:
        pass
    
    @abstractmethod
    def batch_process(self, input_list: List[str], batch_size: int = 32) -> List[np.ndarray]:
        pass
```

#### 3. **BaseMatrix** (matrix/base_matrix.py):
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class BaseMatrix(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.matrix = None
    
    @abstractmethod
    def build_matrix(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def validate_matrix(self) -> bool:
        pass
    
    @abstractmethod
    def save_matrix(self, output_path: str) -> None:
        pass
```

### 🔧 Vantagens da Nova Arquitetura:

1. **Modularidade:**
   - Cada componente tem responsabilidade única
   - Fácil substituição de implementações
   - Testes isolados por módulo

2. **Extensibilidade:**
   - Novos tipos de embeddings facilmente adicionados
   - Suporte a diferentes formatos de matriz
   - Validadores personalizados

3. **Manutenibilidade:**
   - Código organizado por funcionalidade
   - Interfaces claras entre componentes
   - Configuração centralizada

4. **Reutilização:**
   - Componentes podem ser usados independentemente
   - Utilitários compartilhados
   - Pipelines configuráveis

### 📝 Próximos Passos:

1. **Implementar estrutura base** (core/, utils/)
2. **Migrar embeddings** (embeddings/)
3. **Refatorar matrizes** (matrix/)
4. **Reorganizar labels** (labels/)
5. **Implementar validação** (validation/)
6. **Criar pipelines** (pipeline/)
7. **Testes e validação final**

### 🎯 Resultado Esperado:

- **Código 70% mais modular**
- **Redução de duplicação em 60%**
- **Facilidade de testes aumentada em 80%**
- **Tempo de manutenção reduzido em 50%**
- **Extensibilidade aumentada em 90%**
