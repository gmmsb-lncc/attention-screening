"""
Implementação concreta da estratégia para modelos ESM-2.
Contém toda a lógica específica de carregamento, inferência e cleanup para ESM-2.
"""

import os
import gc
from pathlib import Path
from typing import Tuple, Any, Optional
import numpy as np
import torch

from src.build.embeddings.strategies.base_protein_strategy import BaseProteinStrategy
from src.build.core.constants import ESM_MODELS
from src.build.core.exceptions import ModelLoadError, EmbeddingError


class ESM2Strategy(BaseProteinStrategy):
    """
    Estratégia de implementação para modelos ESM-2 (Meta AI).
    
    Características:
    - Suporta modelos de 8M a 15B parâmetros
    - CPU offloading automático para modelos grandes (3B, 15B)
    - Mean pooling sobre sequência (não usa CLS token)
    - Gestão de memória otimizada (gc + empty_cache)
    - Suporta CUDA, MPS (Apple Silicon) e CPU
    
    Responsabilidades:
    - Carregar modelo ESM-2 com configurações apropriadas
    - Gerar embeddings com truncamento automático
    - Gerenciar memória GPU/CPU de forma eficiente
    """
    
    def __init__(self):
        """Inicializa estratégia ESM-2."""
        self.logger = None  # Será configurado quando necessário
        self._cache_dir = None
        self._offload_folder = None
    
    def load(
        self, 
        model_name: str, 
        device: torch.device,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Carrega modelo ESM-2 com CPU offloading opcional.
        
        Args:
            model_name: Nome do modelo (ex: "esm2_t48_15B_UR50D")
            device: Dispositivo PyTorch (cuda/cpu/mps)
            offload_folder: Pasta customizada para offloading (opcional)
            **kwargs: Parâmetros adicionais (logger, etc.)
            
        Returns:
            Tuple (model, alphabet) onde alphabet é o tokenizer do ESM-2
            
        Raises:
            ValueError: Se modelo não for ESM-2 válido
            ModelLoadError: Se falhar ao carregar modelo
        """
        # Configurar logger se fornecido
        self.logger = kwargs.get('logger')
        
        # Validar modelo
        if model_name not in ESM_MODELS:
            raise ValueError(
                f"Modelo ESM-2 '{model_name}' não encontrado. "
                f"Modelos disponíveis: {list(ESM_MODELS.keys())}"
            )
        
        # Configurar cache local
        self._setup_cache_dirs(offload_folder)
        
        # Configurar memória CUDA (nova variável desde PyTorch 2.0)
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Determinar se precisa de CPU offloading
        large_models = ['esm2_t48_15B_UR50D', 'esm2_t36_3B_UR50D']
        needs_offload = model_name in large_models and str(device) == 'cuda'
        
        try:
            if self.logger:
                self.logger.info(f"Carregando modelo ESM-2: {model_name}")
                self.logger.info(f"Cache de modelos: {self._cache_dir}")
            
            # Importar ESM - tentar local primeiro, depois instalado
            esm = self._import_esm()
            
            if needs_offload:
                model, alphabet = self._load_with_offloading(esm, model_name, device)
            else:
                model, alphabet = self._load_standard(esm, model_name, device)
            
            model = model.eval()
            
            if self.logger:
                self.logger.info("✅ Modelo ESM-2 carregado com sucesso")
            
            return model, alphabet
            
        except ImportError as e:
            raise ModelLoadError(
                f"ESM não está disponível. Erro: {e}\n"
                f"Verifique se a pasta ESM/ existe no repositório."
            )
        except Exception as e:
            self._handle_load_error(e, model_name)
    
    def generate(
        self,
        model: Any,
        auxiliary_objects: Any,
        sequence: str,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Gera embedding ESM-2 com mean pooling.
        
        Args:
            model: Modelo ESM-2 carregado
            auxiliary_objects: Alphabet (tokenizer) do ESM-2
            sequence: Sequência de aminoácidos
            device: Dispositivo PyTorch
            **kwargs: Parâmetros opcionais (logger, etc.)
            
        Returns:
            Embedding numpy array (shape: [embedding_dim])
            
        Raises:
            EmbeddingError: Se falhar ao gerar embedding
        """
        alphabet = auxiliary_objects
        self.logger = kwargs.get('logger', self.logger)
        
        # Validar sequência
        if not sequence or not sequence.strip():
            raise EmbeddingError("Sequência vazia")
        
        # Limpar sequência (apenas aminoácidos válidos)
        clean_sequence = ''.join(
            c for c in sequence.upper() 
            if c in 'ACDEFGHIKLMNPQRSTVWY'
        )
        
        if not clean_sequence:
            raise EmbeddingError("Sequência não contém aminoácidos válidos")
        
        # Truncar se necessário
        max_len = self.get_max_length(model.args.arch if hasattr(model, 'args') else model.__class__.__name__)
        
        if len(clean_sequence) > max_len:
            if self.logger:
                self.logger.warning(
                    f"Sequência truncada: {len(clean_sequence)} → {max_len} aa"
                )
            clean_sequence = clean_sequence[:max_len]
        
        try:
            # Preparar tokens
            batch_converter = alphabet.get_batch_converter()
            batch_labels, batch_strs, batch_tokens = batch_converter(
                [("sequence", clean_sequence)]
            )
            batch_tokens = batch_tokens.to(device)
            
            # Inferência
            with torch.no_grad():
                results = model(
                    batch_tokens,
                    repr_layers=[model.num_layers],
                    return_contacts=False
                )
                
                # Extrair embeddings (remover tokens especiais BOS/EOS)
                token_representations = results["representations"][model.num_layers]
                embedding = token_representations[0, 1:-1]  # [seq_len, embed_dim]
                
                # Mean pooling sobre a sequência
                sequence_embedding = embedding.mean(dim=0)  # [embed_dim]
                
                # Mover para CPU
                result = sequence_embedding.cpu().numpy()
            
            # Limpeza de memória CRÍTICA
            del batch_tokens, results, token_representations, embedding, sequence_embedding
            gc.collect()
            
            if str(device) == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return result
            
        except Exception as e:
            # Limpar memória mesmo em erro
            if str(device) == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            raise EmbeddingError(f"Erro ao gerar embedding ESM-2: {e}")
    
    def get_max_length(self, model_name: str) -> int:
        """Retorna comprimento máximo de sequência para o modelo."""
        return ESM_MODELS.get(model_name, {}).get('max_len', 1024)
    
    def get_embedding_dim(self, model_name: str) -> int:
        """Retorna dimensão do embedding do modelo."""
        return ESM_MODELS.get(model_name, {}).get('dim', 1280)
    
    def cleanup(self, model: Any, auxiliary_objects: Any) -> None:
        """
        Libera recursos de memória.
        
        Args:
            model: Modelo a ser limpo (não usado atualmente)
            auxiliary_objects: Alphabet (não usado atualmente)
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== Métodos Privados =====
    
    def _setup_cache_dirs(self, offload_folder: Optional[str] = None) -> None:
        """Configura diretórios de cache e offload."""
        # Cache principal
        self._cache_dir = Path(__file__).parent.parent.parent.parent.parent / "llm" / "models_cache" / "ESM"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['TORCH_HOME'] = str(self._cache_dir)
        
        # Pasta de offload
        if offload_folder:
            self._offload_folder = Path(offload_folder)
        else:
            self._offload_folder = self._cache_dir / "offload"
        self._offload_folder.mkdir(parents=True, exist_ok=True)
    
    def _import_esm(self) -> Any:
        """
        Importa módulo ESM com fallback automático.
        
        Ordem de tentativa:
        1. ESM instalado via pip (fair-esm) - preferido para reprodutibilidade
        2. ESM local em llm/ESM (fallback para desenvolvimento)
        
        Returns:
            Módulo ESM importado
            
        Raises:
            ModelLoadError: Se ESM não estiver disponível
        """
        import sys
        
        # Tentativa 1: ESM instalado via pip (preferido para reprodutibilidade)
        try:
            import esm
            if hasattr(esm, 'pretrained'):
                if self.logger:
                    self.logger.info("✅ ESM loaded from pip package (fair-esm)")
                return esm
        except ImportError:
            pass
        
        # Tentativa 2: ESM local em llm/ESM (fallback)
        # Calcula caminho relativo ao projeto (funciona em qualquer máquina)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        esm_local_path = project_root / "llm" / "ESM"
        
        if esm_local_path.exists():
            esm_str = str(esm_local_path)
            if esm_str not in sys.path:
                sys.path.insert(0, esm_str)
            
            # Forçar reimportação após modificar sys.path
            if 'esm' in sys.modules:
                del sys.modules['esm']
            
            try:
                import esm
                if hasattr(esm, 'pretrained'):
                    if self.logger:
                        self.logger.info(f"✅ ESM loaded from local: {esm_local_path}")
                    return esm
            except ImportError as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Failed to load local ESM: {e}")
        
        # Nenhum ESM disponível - instruções claras para o usuário
        raise ModelLoadError(
            "ESM module not available.\n\n"
            "SOLUTIONS:\n"
            "1. Install via pip (RECOMMENDED):\n"
            "   pip install fair-esm\n\n"
            "2. Or clone the ESM repository:\n"
            "   git clone https://github.com/facebookresearch/esm.git llm/ESM\n\n"
            "For more info: https://github.com/facebookresearch/esm"
        )
    
    def _load_with_offloading(
        self, 
        esm_module: Any, 
        model_name: str, 
        device: torch.device
    ) -> Tuple[Any, Any]:
        """
        Carrega modelo com CPU offloading para modelos grandes.
        
        Args:
            esm_module: Módulo ESM importado
            model_name: Nome do modelo
            device: Dispositivo PyTorch
            
        Returns:
            Tuple (model, alphabet)
        """
        try:
            from accelerate import dispatch_model, infer_auto_device_map
            from accelerate.utils import get_balanced_memory
            
            if self.logger:
                self.logger.info(f"🔄 Modelo grande detectado: {model_name}")
                self.logger.info("🔄 Ativando CPU offloading...")
            
            # Carregar modelo sem mover para device
            model, alphabet = esm_module.pretrained.load_model_and_alphabet(model_name)
            
            # Calcular memória balanceada (20GB GPU + 30GB CPU)
            max_memory = get_balanced_memory(
                model,
                max_memory={0: "20GiB", "cpu": "30GiB"},
                no_split_module_classes=["TransformerLayer"]
            )
            
            # Criar device map automático
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["TransformerLayer"]
            )
            
            # Dispatch para múltiplos devices
            model = dispatch_model(
                model,
                device_map=device_map,
                offload_folder=str(self._offload_folder)
            )
            
            if self.logger:
                self.logger.info("✅ CPU offloading ativado")
                self.logger.info(f"   Offload folder: {self._offload_folder}")
            
            return model, alphabet
            
        except ImportError:
            if self.logger:
                self.logger.warning("⚠️  accelerate não encontrado. Carregando sem offloading...")
            return self._load_standard(esm_module, model_name, device)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️  Falha no offloading: {e}")
                self.logger.warning("   Tentando carregamento padrão...")
            return self._load_standard(esm_module, model_name, device)
    
    def _load_standard(
        self, 
        esm_module: Any, 
        model_name: str, 
        device: torch.device
    ) -> Tuple[Any, Any]:
        """
        Carregamento padrão sem offloading.
        
        Args:
            esm_module: Módulo ESM importado
            model_name: Nome do modelo
            device: Dispositivo PyTorch
            
        Returns:
            Tuple (model, alphabet)
        """
        model, alphabet = esm_module.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device)
        return model, alphabet
    
    def _handle_load_error(self, error: Exception, model_name: str) -> None:
        """
        Trata erros de carregamento de forma específica.
        
        Args:
            error: Exceção capturada
            model_name: Nome do modelo
            
        Raises:
            ModelLoadError: Com mensagem apropriada
        """
        error_msg = str(error).lower()
        
        # Erro 404 / arquivo não encontrado
        if "404" in error_msg or "not found" in error_msg or "could not load" in error_msg:
            raise ModelLoadError(
                f"Modelo ESM-2 '{model_name}' não está disponível.\n"
                f"URL esperada: https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt\n"
                f"Erro: {error}\n\n"
                f"SOLUÇÕES:\n"
                f"1. Verifique o nome do modelo\n"
                f"2. Modelo 15B pode não estar público (baixe manualmente)\n"
                f"3. Use modelo menor: esm2_t36_3B_UR50D ou esm2_t33_650M_UR50D"
            )
        
        # Erro de memória
        elif "out of memory" in error_msg or "oom" in error_msg:
            raise ModelLoadError(
                f"Memória insuficiente para '{model_name}'.\n"
                f"Erro: {error}\n\n"
                f"SOLUÇÕES:\n"
                f"1. Use modelo menor\n"
                f"2. Aumente memória disponível\n"
                f"3. Instale accelerate para CPU offloading"
            )
        
        # Erro genérico
        else:
            raise ModelLoadError(
                f"Falha ao carregar modelo ESM-2 '{model_name}'.\n"
                f"Cache: {self._cache_dir}\n"
                f"Erro: {type(error).__name__}: {error}"
            )
