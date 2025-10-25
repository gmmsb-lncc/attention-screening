#!/usr/bin/env python3
"""
Pipeline Completo DockTKinase
==============================

Pipeline end-to-end para:
1. Carregar datasets de kinases
2. Gerar embeddings com ESM-2
3. Estratificar dataset de forma robusta
4. Treinar e avaliar classificador

Uso:
    python run_complete_pipeline.py --dataset human --model esm2_t6_8M_UR50D --test-size 0.2
    
Datasets disponíveis:
    - human: kinase_human_compounds.tsv (404 MB)
    - non_human: kinase_non_human_compounds.tsv (11 MB)
    - all: kinase_all_compounds.tsv (415 MB)
"""

import sys
import argparse
from pathlib import Path
import time
import json
from datetime import datetime
import gc

# Visualização
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar paths
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / 'src'))
ESM_PATH = ROOT_DIR / 'ESM'
sys.path.insert(0, str(ESM_PATH))

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    accuracy_score, precision_recall_fscore_support
)
from scipy import stats


class CompletePipeline:
    """Pipeline completo de processamento e classificação"""
    
    def __init__(
        self, 
        dataset_name='human',
        esm_model='esm2_t6_8M_UR50D',
        val_size=0.1,
        test_size=0.1,
        random_state=42,
        max_samples=None,
        device='auto',
        output_dir='pipeline_output',
        label_method='pchembl',
        label_threshold=None,
        verbose=True
    ):
        """
        Inicializar pipeline
        
        Args:
            dataset_name: 'human', 'non_human' ou 'all'
            esm_model: Modelo ESM-2 a usar
            val_size: Proporção do conjunto de validação (default: 0.1 = 10%)
            test_size: Proporção do conjunto de teste (default: 0.1 = 10%)
            random_state: Seed para reprodutibilidade
            max_samples: Limite de amostras (None = todas)
            device: 'cpu', 'cuda', ou 'auto'
            output_dir: Diretório para salvar resultados
            label_method: Método para criar labels - 'pchembl', 'kd', 'ki', 'ic50', 'auto'
            label_threshold: Threshold para labels (None = usar padrão)
            verbose: Mostrar logs detalhados
        """
        self.dataset_name = dataset_name
        self.esm_model = esm_model
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.max_samples = max_samples
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.label_method = label_method.lower()
        self.label_threshold = label_threshold
        self.verbose = verbose
        
        # Criar diretório de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Estatísticas do pipeline
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'dataset': dataset_name,
            'model': esm_model,
            'device': str(self.device),
            'label_method': label_method,
            'label_threshold': label_threshold
        }
        
        if self.verbose:
            self._print_header()
    
    def _setup_device(self, device):
        """Configurar device (CPU/GPU)"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _print_header(self):
        """Imprimir cabeçalho"""
        print('='*60)
        print('🚀 PIPELINE COMPLETO DOCKTKINASE')
        print('='*60)
        print(f'📊 Dataset: {self.dataset_name}')
        print(f'🧬 Modelo ESM-2: {self.esm_model}')
        print(f'💻 Device: {self.device}')
        print(f'📁 Output: {self.output_dir}')
        print(f'🏷️  Labels: {self.label_method} (threshold: {self.label_threshold})')
        print(f'🔀 Split: {(1-self.val_size-self.test_size)*100:.0f}% treino, {self.val_size*100:.0f}% val, {self.test_size*100:.0f}% teste')
        print('='*60)
        print()
    
    def load_dataset(self):
        """
        Carregar dataset
        
        Returns:
            pd.DataFrame com dados carregados
        """
        if self.verbose:
            print('📂 ETAPA 1: Carregando Dataset')
            print('-'*60)
        
        # Mapear nome do dataset para arquivo
        dataset_files = {
            'human': 'kinase_human_compounds.tsv',
            'non_human': 'kinase_non_human_compounds.tsv',
            'all': 'kinase_all_compounds.tsv'
        }
        
        if self.dataset_name not in dataset_files:
            raise ValueError(
                f"Dataset '{self.dataset_name}' inválido. "
                f"Use: {list(dataset_files.keys())}"
            )
        
        dataset_path = ROOT_DIR / 'tests' / 'datasets' / dataset_files[self.dataset_name]
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
        
        start_time = time.time()
        
        # Carregar dataset
        if self.verbose:
            print(f'   Carregando {dataset_path.name}...')
        
        df = pd.read_csv(dataset_path, sep='\t')
        
        # Limitar amostras se necessário
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=self.random_state).reset_index(drop=True)
            if self.verbose:
                print(f'   ⚠️  Limitado a {self.max_samples} amostras')
        
        load_time = time.time() - start_time
        
        if self.verbose:
            print(f'   ✅ Dataset carregado: {len(df):,} amostras')
            print(f'   ⏱️  Tempo: {load_time:.2f}s')
            print(f'   📊 Colunas: {list(df.columns)}')
            print()
        
        self.stats['load_time'] = load_time
        self.stats['total_samples'] = len(df)
        self.stats['columns'] = list(df.columns)
        
        return df
    
    def generate_embeddings(self, df, batch_size=8):
        """
        Gerar embeddings ESM-2 com gerenciamento inteligente de memória
        
        Features:
        - Detecção automática de memória GPU
        - Ajuste dinâmico de batch size
        - Limpeza agressiva de memória
        - Checkpoint para recuperação
        - Suporte a sequências longas
        
        Args:
            df: DataFrame com coluna 'seq'
            batch_size: Tamanho inicial do batch (será ajustado automaticamente)
            
        Returns:
            np.ndarray com embeddings
        """
        if self.verbose:
            print('🧬 ETAPA 2: Gerando Embeddings ESM-2')
            print('-'*60)
        
        import esm
        
        start_time = time.time()
        
        # Carregar modelo
        if self.verbose:
            print(f'   Carregando modelo {self.esm_model}...')
        
        model_func = getattr(esm.pretrained, self.esm_model)
        model, alphabet = model_func()
        model = model.to(self.device)
        model.eval()
        
        batch_converter = alphabet.get_batch_converter()
        
        load_time = time.time() - start_time
        
        if self.verbose:
            print(f'   ✅ Modelo carregado em {load_time:.2f}s')
            print(f'   📊 Gerando embeddings para {len(df):,} sequências...')
        
        # =====================================================================
        # SISTEMA DE BATCH INTELIGENTE COM GERENCIAMENTO DE MEMÓRIA
        # =====================================================================
        
        embeddings = []
        sequences = df['seq'].tolist()
        
        # Configuração inicial
        current_batch_size = batch_size
        min_batch_size = 1
        max_retries = 3
        checkpoint_interval = 100  # Salvar checkpoint a cada N batches
        
        # Estatísticas
        oom_count = 0
        total_batches = 0
        
        # Arquivo checkpoint
        checkpoint_file = Path('tmp') / 'embedding_checkpoint.npz'
        checkpoint_file.parent.mkdir(exist_ok=True)
        
        # Tentar carregar checkpoint anterior
        start_idx = 0
        if checkpoint_file.exists():
            try:
                checkpoint = np.load(checkpoint_file, allow_pickle=True)
                embeddings = list(checkpoint['embeddings'])
                start_idx = checkpoint['last_idx']
                if self.verbose:
                    print(f'   🔄 Checkpoint encontrado! Retomando do índice {start_idx}')
            except:
                pass
        
        i = start_idx
        while i < len(sequences):
            try:
                # Batch atual
                batch_end = min(i + current_batch_size, len(sequences))
                batch_seqs = sequences[i:batch_end]
                batch_data = [(f'seq_{j}', seq) for j, seq in enumerate(batch_seqs)]
                
                # Converter batch
                _, _, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(self.device)
                
                # Gerar embeddings
                with torch.no_grad():
                    # Limpar cache antes
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    results = model(batch_tokens, repr_layers=[model.num_layers])
                    
                    # Média sobre posições (sem tokens especiais)
                    token_embeddings = results['representations'][model.num_layers]
                    batch_embeddings = token_embeddings[:, 1:-1, :].mean(1)
                    
                    # Mover para CPU imediatamente
                    batch_embeddings_cpu = batch_embeddings.cpu().numpy()
                
                embeddings.append(batch_embeddings_cpu)
                
                # Liberar memória GPU imediatamente
                del batch_tokens, results, token_embeddings, batch_embeddings, batch_embeddings_cpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Garbage collection agressivo
                gc.collect()
                
                # Progresso
                if self.verbose and (i + current_batch_size) % 50 == 0:
                    processed = min(i + current_batch_size, len(sequences))
                    pct = 100 * processed / len(sequences)
                    
                    # Mostrar uso de memória se CUDA disponível
                    mem_info = ""
                    if torch.cuda.is_available():
                        mem_used = torch.cuda.memory_allocated(0) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                        mem_info = f" | GPU: {mem_used:.1f}GB usado, {mem_reserved:.1f}GB reservado"
                    
                    print(f'      {processed:,}/{len(sequences):,} ({pct:.1f}%) | batch={current_batch_size}{mem_info}')
                
                # Salvar checkpoint periodicamente
                total_batches += 1
                if total_batches % checkpoint_interval == 0:
                    np.savez_compressed(
                        checkpoint_file,
                        embeddings=np.vstack(embeddings),
                        last_idx=batch_end
                    )
                
                # Sucesso! Tentar aumentar batch size gradualmente
                if oom_count == 0 and current_batch_size < batch_size * 2:
                    current_batch_size = min(current_batch_size + 1, batch_size * 2)
                
                # Avançar
                i = batch_end
                
            except torch.cuda.OutOfMemoryError as e:
                oom_count += 1
                
                if self.verbose:
                    print(f'   ⚠️  OOM Error! Reduzindo batch size de {current_batch_size}...')
                
                # Limpar toda a memória
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Reduzir batch size pela metade
                current_batch_size = max(min_batch_size, current_batch_size // 2)
                
                if current_batch_size < min_batch_size:
                    # Batch mínimo ainda falha - tentar com variáveis de ambiente
                    if self.verbose:
                        print('   ❌ Falha mesmo com batch=1. Tentando com PYTORCH_CUDA_ALLOC_CONF...')
                    
                    import os
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    
                    # Resetar modelo
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    model_func = getattr(esm.pretrained, self.esm_model)
                    model, alphabet = model_func()
                    model = model.to(self.device)
                    model.eval()
                    batch_converter = alphabet.get_batch_converter()
                    
                    current_batch_size = min_batch_size
                
                if oom_count > max_retries:
                    raise RuntimeError(
                        f"Falha persistente de memória após {max_retries} tentativas. "
                        f"Considere:\n"
                        f"1. Usar modelo menor (ex: esm2_t33_650M_UR50D)\n"
                        f"2. Reduzir tamanho do dataset com --max-samples\n"
                        f"3. Usar CPU com --device cpu (muito mais lento)\n"
                        f"4. Aumentar memória GPU disponível"
                    )
                
                # Não avançar o índice - tentar novamente com batch menor
                continue
        
        # Combinar todos os embeddings
        embeddings = np.vstack(embeddings)
        
        # Remover checkpoint ao finalizar
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        embed_time = time.time() - start_time - load_time
        
        if self.verbose:
            print(f'   ✅ Embeddings gerados!')
            print(f'   📊 Shape: {embeddings.shape}')
            print(f'   ⏱️  Tempo: {embed_time:.2f}s ({embed_time/len(df):.3f}s/seq)')
            if oom_count > 0:
                print(f'   ⚠️  OOM Errors: {oom_count} (batch ajustado automaticamente)')
            print()
        
        self.stats['embedding_time'] = embed_time
        self.stats['embedding_shape'] = embeddings.shape
        self.stats['embedding_dim'] = embeddings.shape[1]
        self.stats['oom_errors'] = oom_count
        
        return embeddings
    
    def create_labels(self, df):
        """
        Criar labels baseado no método escolhido
        
        PRIORIDADE: Valores de atividade (Kd, Ki, IC50) > pchembl_value
        Threshold padrão: <= 1000 nM = ATIVO (classe 1)
                         > 1000 nM = INATIVO (classe 0)
        
        Args:
            df: DataFrame com dados
            
        Returns:
            np.ndarray com labels (0=inativo/negativo, 1=ativo/positivo)
        """
        if self.verbose:
            print('🏷️  CRIANDO LABELS')
            print('-'*60)
        
        # Se já existe coluna 'label', usar ela
        if 'label' in df.columns:
            if self.verbose:
                print('   ✅ Coluna "label" encontrada, usando labels existentes')
            return df['label'].values
        
        # Determinar método e threshold
        method = self.label_method
        threshold = self.label_threshold
        
        # Método automático: PRIORIZAR valores de atividade sobre pchembl
        # No modo AUTO, usar TODAS as medidas (Ki + Kd + IC50) combinadas
        if method == 'auto':
            if 'standard_value' in df.columns and 'standard_type' in df.columns:
                # Verificar se há alguma medida de atividade disponível
                valid_types = ['Ki', 'Kd', 'IC50']
                has_activity = df['standard_type'].isin(valid_types).any()
                if has_activity:
                    method = 'combined'  # Novo método interno para usar todas as medidas
                else:
                    # Fallback para pchembl se não houver valores de atividade
                    if 'pchembl_value' in df.columns and df['pchembl_value'].notna().sum() > 0:
                        method = 'pchembl'
            elif 'pchembl_value' in df.columns and df['pchembl_value'].notna().sum() > 0:
                method = 'pchembl'
        
        if self.verbose:
            display_method = 'Ki+Kd+IC50 (combinado)' if method == 'combined' else method
            print(f'   📊 Método selecionado: {display_method}')
        
        # Criar labels baseado no método
        if method == 'pchembl':
            if 'pchembl_value' not in df.columns:
                raise ValueError("Coluna 'pchembl_value' não encontrada no dataset")
            
            # Threshold padrão: pchembl >= 6 (equivalente a IC50 <= 1000 nM)
            if threshold is None:
                threshold = 6.0
            
            # pchembl_value MAIOR = MAIS ATIVO (positivo)
            # pchembl = -log10(IC50 in M), então pchembl >= 6 significa IC50 <= 1 µM
            labels = (df['pchembl_value'] >= threshold).astype(int)
            
            if self.verbose:
                print(f'   ✅ Labels criados: pchembl_value >= {threshold} = ATIVO (classe 1)')
                print(f'                      pchembl_value < {threshold} = INATIVO (classe 0)')
        
        elif method in ['kd', 'ki', 'ic50']:
            if 'standard_value' not in df.columns or 'standard_type' not in df.columns:
                raise ValueError("Colunas 'standard_value' e 'standard_type' não encontradas")
            
            # Threshold padrão: <= 1000 nM = ativo
            if threshold is None:
                threshold = 1000.0
            
            # USAR APENAS O TIPO ESPECÍFICO solicitado (ic50, ki ou kd)
            type_map = {'kd': 'Kd', 'ki': 'Ki', 'ic50': 'IC50'}
            std_type = type_map[method]
            
            # Valores MENORES = MAIS ATIVOS (positivo)
            # Kd/Ki/IC50 <= 1000 nM = bom (classe 1)
            labels = np.zeros(len(df), dtype=int)
            mask = df['standard_type'] == std_type
            labels[mask] = (df.loc[mask, 'standard_value'] <= threshold).astype(int)
            
            # Remover amostras sem o tipo de medida desejado
            valid_mask = df['standard_type'] == std_type
            n_removed = (~valid_mask).sum()
            
            if self.verbose:
                print(f'   ✅ Labels criados: {std_type} <= {threshold} nM = ATIVO (classe 1)')
                print(f'                      {std_type} > {threshold} nM = INATIVO (classe 0)')
                if n_removed > 0:
                    print(f'   ⚠️  {n_removed} amostras removidas (sem medida {std_type})')
            
            # Filtrar dataframe
            if n_removed > 0:
                df = df[valid_mask].copy().reset_index(drop=True)
                labels = labels[valid_mask]
        
        elif method == 'combined':
            # Modo COMBINADO (usado por 'auto'): Usar TODAS as medidas de atividade (IC50 + Ki + Kd)
            if 'standard_value' not in df.columns or 'standard_type' not in df.columns:
                raise ValueError("Colunas 'standard_value' e 'standard_type' não encontradas")
            
            # Threshold padrão: <= 1000 nM = ativo
            if threshold is None:
                threshold = 1000.0
            
            # Usar TODAS as medidas de atividade: Ki, Kd, IC50 (em ordem de prioridade)
            valid_types = ['Ki', 'Kd', 'IC50']
            valid_mask = df['standard_type'].isin(valid_types)
            
            # Valores MENORES = MAIS ATIVOS (positivo)
            # Ki/Kd/IC50 <= 1000 nM = ATIVO (classe 1)
            labels = np.zeros(len(df), dtype=int)
            labels[valid_mask] = (df.loc[valid_mask, 'standard_value'] <= threshold).astype(int)
            
            # Estatísticas por tipo de medida (ordem de prioridade: Ki > Kd > IC50)
            type_counts = df[valid_mask]['standard_type'].value_counts()
            n_removed = (~valid_mask).sum()
            
            if self.verbose:
                print(f'   ✅ Labels criados usando TODAS as medidas de atividade:')
                print(f'      Ki/Kd/IC50 <= {threshold} nM = ATIVO (classe 1)')
                print(f'      Ki/Kd/IC50 > {threshold} nM = INATIVO (classe 0)')
                print(f'\n   📊 Distribuição por tipo de medida (prioridade: Ki > Kd > IC50):')
                for measure_type in ['Ki', 'Kd', 'IC50']:
                    count = type_counts.get(measure_type, 0)
                    pct = (count / valid_mask.sum() * 100) if valid_mask.sum() > 0 else 0
                    print(f'      {measure_type}: {count:,} amostras ({pct:.1f}%)')
                if n_removed > 0:
                    print(f'   ⚠️  {n_removed:,} amostras removidas (sem medida Ki/Kd/IC50)')
            
            # Filtrar dataframe para manter apenas amostras com medidas válidas
            if n_removed > 0:
                df = df[valid_mask].copy().reset_index(drop=True)
                labels = labels[valid_mask]
        
        else:
            raise ValueError(
                f"Método '{method}' inválido. Use: 'pchembl', 'kd', 'ki', 'ic50', 'auto'"
            )
        
        # Estatísticas dos labels
        unique, counts = np.unique(labels, return_counts=True)
        
        if self.verbose:
            print(f'\n   📊 Distribuição de labels:')
            for label, count in zip(unique, counts):
                label_name = 'ATIVO/POSITIVO' if label == 1 else 'INATIVO/NEGATIVO'
                print(f'      Classe {label} ({label_name}): {count:,} ({count/len(labels)*100:.2f}%)')
            print()
        
        self.stats['label_creation'] = {
            'method': method,
            'threshold': threshold,
            'class_0_count': int(counts[0]) if 0 in unique else 0,
            'class_1_count': int(counts[1]) if 1 in unique else 0,
            'total': len(labels)
        }
        
        return labels, df
    
    def stratified_split(self, X, y):
        """
        Divisão estratificada em 3 conjuntos: treino, validação e teste
        SEM DATA LEAKING
        
        Args:
            X: Features (embeddings)
            y: Labels
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test
        """
        if self.verbose:
            print('🔀 ETAPA 3: Divisão Estratificada (Treino/Val/Teste)')
            print('-'*60)
        
        start_time = time.time()
        
        # Criar array de índices para rastrear amostras originais
        indices = np.arange(len(y))
        
        # Distribuição original
        unique, counts = np.unique(y, return_counts=True)
        
        if self.verbose:
            print(f'   📊 Distribuição original:')
            for label, count in zip(unique, counts):
                label_name = 'ATIVO' if label == 1 else 'INATIVO'
                print(f'      Classe {label} ({label_name}): {count:,} ({count/len(y)*100:.2f}%)')
        
        # PASSO 1: Separar TEST set primeiro (10%)
        # Isso garante que o test set nunca foi visto durante treinamento/validação
        test_size_adjusted = self.test_size
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X, y, indices,
            test_size=test_size_adjusted,
            stratify=y,
            random_state=self.random_state
        )
        
        # PASSO 2: Do restante (90%), separar VALIDATION (10% do total = 11.1% do restante)
        # val_size_adjusted calcula a proporção correta do restante
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        split_time = time.time() - start_time
        
        # Validar proporções em cada conjunto
        def get_props(y_subset):
            return np.array([np.sum(y_subset == label) / len(y_subset) for label in unique])
        
        train_props = get_props(y_train)
        val_props = get_props(y_val)
        test_props = get_props(y_test)
        
        # Calcular diferenças máximas
        max_diff_train_val = np.max(np.abs(train_props - val_props))
        max_diff_train_test = np.max(np.abs(train_props - test_props))
        max_diff_val_test = np.max(np.abs(val_props - test_props))
        max_diff_overall = max(max_diff_train_val, max_diff_train_test, max_diff_val_test)
        
        # Testes chi-quadrado
        train_counts = np.array([np.sum(y_train == label) for label in unique])
        val_counts = np.array([np.sum(y_val == label) for label in unique])
        test_counts = np.array([np.sum(y_test == label) for label in unique])
        
        total = len(y)
        expected_train = counts * (len(y_train) / total)
        expected_val = counts * (len(y_val) / total)
        expected_test = counts * (len(y_test) / total)
        
        chi2_train = np.sum((train_counts - expected_train)**2 / expected_train)
        chi2_val = np.sum((val_counts - expected_val)**2 / expected_val)
        chi2_test = np.sum((test_counts - expected_test)**2 / expected_test)
        
        df_chi = len(unique) - 1
        p_value_train = 1 - stats.chi2.cdf(chi2_train, df_chi)
        p_value_val = 1 - stats.chi2.cdf(chi2_val, df_chi)
        p_value_test = 1 - stats.chi2.cdf(chi2_test, df_chi)
        
        if self.verbose:
            print(f'\n   ✅ Split realizado (SEM data leaking):')
            print(f'      Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)')
            print(f'      Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)')
            print(f'      Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)')
            
            print(f'\n   📊 Proporções por conjunto:')
            print(f'      Train: ', end='')
            for label, prop in zip(unique, train_props):
                label_name = 'ATIVO' if label == 1 else 'INATIVO'
                print(f'{label_name}={prop*100:.2f}%  ', end='')
            
            print(f'\n      Val:   ', end='')
            for label, prop in zip(unique, val_props):
                label_name = 'ATIVO' if label == 1 else 'INATIVO'
                print(f'{label_name}={prop*100:.2f}%  ', end='')
            
            print(f'\n      Test:  ', end='')
            for label, prop in zip(unique, test_props):
                label_name = 'ATIVO' if label == 1 else 'INATIVO'
                print(f'{label_name}={prop*100:.2f}%  ', end='')
            
            print(f'\n\n   📏 Diferenças de proporções:')
            print(f'      Train-Val:  {max_diff_train_val*100:.4f}%')
            print(f'      Train-Test: {max_diff_train_test*100:.4f}%')
            print(f'      Val-Test:   {max_diff_val_test*100:.4f}%')
            print(f'      Máxima:     {max_diff_overall*100:.4f}%')
            
            print(f'\n   🧪 Testes Chi-quadrado:')
            print(f'      Train: p={p_value_train:.4f}')
            print(f'      Val:   p={p_value_val:.4f}')
            print(f'      Test:  p={p_value_test:.4f}')
            
            if max_diff_overall < 0.05 and all(p > 0.05 for p in [p_value_train, p_value_val, p_value_test]):
                print(f'\n   ✅ Split validado estatisticamente!')
                print(f'   ✅ Sem data leaking (test separado primeiro)')
            else:
                print(f'\n   ⚠️  Split pode não estar perfeitamente balanceado')
            
            print(f'\n   ⏱️  Tempo: {split_time:.2f}s')
            print()
        
        self.stats['split_time'] = split_time
        self.stats['train_size'] = len(X_train)
        self.stats['val_size'] = len(X_val)
        self.stats['test_size'] = len(X_test)
        self.stats['train_samples'] = len(X_train)
        self.stats['val_samples'] = len(X_val)
        self.stats['test_samples'] = len(X_test)
        self.stats['max_proportion_diff'] = float(max_diff_overall)
        self.stats['chi2_p_values'] = {
            'train': float(p_value_train),
            'val': float(p_value_val),
            'test': float(p_value_test)
        }
        
        # 🔥 NOVO: Salvar índices dos splits para reutilização no pipeline de regressão
        self.stats['split_indices'] = {
            'train': idx_train.tolist(),
            'val': idx_val.tolist(),
            'test': idx_test.tolist()
        }
        
        # Gerar visualização da estratificação
        if self.verbose:
            self.plot_stratification(y_train, y_val, y_test, y)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test
    
    def plot_stratification(self, y_train, y_val, y_test, y_original):
        """
        Criar visualizações da estratificação
        
        Args:
            y_train: Labels de treino
            y_val: Labels de validação
            y_test: Labels de teste
            y_original: Labels originais
        """
        try:
            # Criar diretório de visualizações
            viz_dir = self.output_dir / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Configurar estilo
            sns.set_style("whitegrid")
            plt.rcParams['figure.facecolor'] = 'white'
            
            # Criar figura com 3 subplots
            fig = plt.figure(figsize=(16, 5))
            
            # ===== SUBPLOT 1: Contagem de classes =====
            ax1 = plt.subplot(1, 3, 1)
            
            datasets = ['Original', 'Train', 'Val', 'Test']
            y_sets = [y_original, y_train, y_val, y_test]
            
            class_0_counts = [np.sum(y == 0) for y in y_sets]
            class_1_counts = [np.sum(y == 1) for y in y_sets]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, class_0_counts, width, label='Classe 0 (INATIVO)', 
                           color='#3498db', alpha=0.8)
            bars2 = ax1.bar(x + width/2, class_1_counts, width, label='Classe 1 (ATIVO)', 
                           color='#e74c3c', alpha=0.8)
            
            ax1.set_xlabel('Conjunto de Dados', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Número de Amostras', fontsize=12, fontweight='bold')
            ax1.set_title('Distribuição de Classes por Conjunto', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(datasets)
            ax1.legend(loc='upper right')
            ax1.grid(axis='y', alpha=0.3)
            
            # Adicionar valores nas barras
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # ===== SUBPLOT 2: Proporções =====
            ax2 = plt.subplot(1, 3, 2)
            
            proportions_0 = [c / (c + class_1_counts[i]) * 100 
                           for i, c in enumerate(class_0_counts)]
            proportions_1 = [c / (class_0_counts[i] + c) * 100 
                           for i, c in enumerate(class_1_counts)]
            
            bars1 = ax2.bar(x - width/2, proportions_0, width, label='Classe 0 (INATIVO)',
                           color='#3498db', alpha=0.8)
            bars2 = ax2.bar(x + width/2, proportions_1, width, label='Classe 1 (ATIVO)',
                           color='#e74c3c', alpha=0.8)
            
            ax2.set_xlabel('Conjunto de Dados', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Proporção (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Proporções de Classes (Estratificação)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(datasets)
            ax2.legend(loc='upper right')
            ax2.set_ylim([0, 100])
            ax2.grid(axis='y', alpha=0.3)
            
            # Adicionar valores nas barras
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # ===== SUBPLOT 3: Tamanhos dos conjuntos =====
            ax3 = plt.subplot(1, 3, 3)
            
            sizes = [len(y) for y in y_sets]
            colors = ['#95a5a6', '#2ecc71', '#f39c12', '#9b59b6']
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=datasets, autopct='%1.1f%%',
                                               colors=colors, startangle=90,
                                               textprops={'fontsize': 11, 'fontweight': 'bold'})
            
            ax3.set_title('Distribuição de Amostras por Conjunto', 
                         fontsize=14, fontweight='bold')
            
            # Adicionar legenda com contagens
            legend_labels = [f'{ds}: {sz:,} amostras' for ds, sz in zip(datasets, sizes)]
            ax3.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
            
            plt.tight_layout()
            
            # Salvar figura
            viz_file = viz_dir / 'stratification_analysis.png'
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f'   📊 Visualização salva: {viz_file}')
        
        except Exception as e:
            if self.verbose:
                print(f'   ⚠️  Erro ao gerar visualização: {e}')
    
    def train_classifier(self, X_train, y_train, X_val=None, y_val=None):
        """
        Treinar classificador
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            X_val: Features de validação (opcional)
            y_val: Labels de validação (opcional)
            
        Returns:
            Modelo treinado
        """
        if self.verbose:
            print('🤖 ETAPA 4: Treinamento do Classificador')
            print('-'*60)
        
        from sklearn.ensemble import RandomForestClassifier
        
        start_time = time.time()
        
        if self.verbose:
            print(f'   Treinando Random Forest...')
        
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        clf.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Calcular métricas completas no treino
        y_train_pred = clf.predict(X_train)
        y_train_proba = clf.predict_proba(X_train)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            y_train, y_train_pred, average='weighted'
        )
        
        # ROC AUC para treino
        if len(np.unique(y_train)) == 2:
            train_roc_auc = roc_auc_score(y_train, y_train_proba[:, 1])
        else:
            train_roc_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='weighted')
        
        if self.verbose:
            print(f'   ✅ Modelo treinado!')
            print(f'   📊 N° estimadores: {clf.n_estimators}')
            print(f'   📊 Max depth: {clf.max_depth}')
            print(f'   📊 Métricas Treino:')
            print(f'      Acurácia: {train_acc:.4f}')
            print(f'      F1-Score: {train_f1:.4f}')
            print(f'      Precisão: {train_precision:.4f}')
            print(f'      Recall: {train_recall:.4f}')
            print(f'      ROC-AUC: {train_roc_auc:.4f}')
            
            if X_val is not None and y_val is not None:
                val_acc = clf.score(X_val, y_val)
                print(f'   📊 Acurácia Validação: {val_acc:.4f}')
                self.stats['val_accuracy'] = float(val_acc)
            
            print(f'   ⏱️  Tempo: {train_time:.2f}s')
            print()
        
        # Salvar todas as métricas de treino
        self.stats['train_time'] = train_time
        self.stats['train_metrics'] = {
            'accuracy': float(train_acc),
            'f1': float(train_f1),
            'precision': float(train_precision),
            'recall': float(train_recall),
            'roc_auc': float(train_roc_auc)
        }
        self.stats['model_type'] = 'RandomForest'
        self.stats['n_estimators'] = clf.n_estimators
        
        return clf
    
    def evaluate_classifier(self, clf, X_test, y_test, dataset_name='Test', df_subset=None, indices=None):
        """
        Avaliar classificador
        
        Args:
            clf: Modelo treinado
            X_test: Features de teste
            y_test: Labels de teste
            dataset_name: Nome do conjunto ('Test' ou 'Validation')
            df_subset: DataFrame com informações originais (opcional)
            indices: Índices originais das amostras (opcional)
            
        Returns:
            Dict com métricas
        """
        if self.verbose:
            print(f'📊 ETAPA 5: Avaliação do Classificador ({dataset_name})')
            print('-'*60)
        
        start_time = time.time()
        
        # Predições
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # ROC AUC (se binário)
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            avg_precision = average_precision_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            avg_precision = None
        
        eval_time = time.time() - start_time
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision) if avg_precision else None
        }
        
        if self.verbose:
            print(f'   📈 Métricas:')
            print(f'      Accuracy: {accuracy:.4f}')
            print(f'      Precision: {precision:.4f}')
            print(f'      Recall: {recall:.4f}')
            print(f'      F1-Score: {f1:.4f}')
            print(f'      ROC AUC: {roc_auc:.4f}')
            if avg_precision:
                print(f'      Avg Precision: {avg_precision:.4f}')
            
            # Confusion matrix
            print(f'\n   🔢 Matriz de Confusão:')
            cm = confusion_matrix(y_test, y_pred)
            print(f'{cm}')
            
            # Classification report
            print(f'\n   📋 Relatório de Classificação:')
            print(classification_report(y_test, y_pred))
            
            print(f'   ⏱️  Tempo: {eval_time:.2f}s')
            print()
        
        self.stats['eval_time'] = eval_time
        self.stats['metrics'] = metrics
        
        # Salvar CSV detalhado das predições
        if df_subset is not None and indices is not None:
            self.save_predictions_csv(y_test, y_pred, y_proba, df_subset, indices, dataset_name)
        
        # Gerar visualização da avaliação
        if self.verbose:
            self.plot_evaluation(y_test, y_pred, y_proba, dataset_name)
        
        return metrics
    
    def save_predictions_csv(self, y_true, y_pred, y_proba, df_subset, indices, dataset_name='Test'):
        """
        Salvar CSV detalhado com informações de cada predição
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            y_proba: Probabilidades preditas
            df_subset: DataFrame original com informações dos compostos
            indices: Índices das amostras
            dataset_name: Nome do conjunto (Test/Validation)
        """
        def safe_get(row_dict, key, default='N/A'):
            """Obter valor do dicionário tratando NaN e None"""
            value = row_dict.get(key, default)
            if pd.isna(value):
                return default
            return value
        
        try:
            # Determinar categoria de predição (TP, FP, TN, FN)
            categories = []
            for yt, yp in zip(y_true, y_pred):
                if yt == 1 and yp == 1:
                    categories.append('TP')  # True Positive
                elif yt == 0 and yp == 1:
                    categories.append('FP')  # False Positive
                elif yt == 0 and yp == 0:
                    categories.append('TN')  # True Negative
                else:  # yt == 1 and yp == 0
                    categories.append('FN')  # False Negative
            
            # Preparar dados para o CSV
            predictions_data = []
            
            for idx, (i, cat, yt, yp) in enumerate(zip(indices, categories, y_true, y_pred)):
                row_data = df_subset.iloc[i].to_dict()
                
                prediction_row = {
                    'prediction_category': cat,
                    'molregno': safe_get(row_data, 'molregno'),
                    'seq_id': safe_get(row_data, 'seq_id'),
                    'target_kinase': safe_get(row_data, 'target_kinase'),
                    'canonical_smiles': safe_get(row_data, 'canonical_smiles'),
                    'aminoacid_sequence': safe_get(row_data, 'seq'),
                    'true_label': int(yt),
                    'predicted_label': int(yp),
                    'probability_class_0': float(y_proba[idx, 0]),
                    'probability_class_1': float(y_proba[idx, 1]),
                    'pchembl_value': safe_get(row_data, 'pchembl_value'),
                    'standard_value': safe_get(row_data, 'standard_value'),
                    'standard_type': safe_get(row_data, 'standard_type'),
                    'compound_name': safe_get(row_data, 'compound_name'),
                    'chembl_id': safe_get(row_data, 'chembl_id'),
                    'organism': safe_get(row_data, 'organism'),
                    'dataset': dataset_name
                }
                predictions_data.append(prediction_row)
            
            # Criar DataFrame e salvar
            df_predictions = pd.DataFrame(predictions_data)
            
            # Ordenar por categoria para facilitar análise
            category_order = {'TP': 0, 'FP': 1, 'TN': 2, 'FN': 3}
            df_predictions['sort_key'] = df_predictions['prediction_category'].map(category_order)
            df_predictions = df_predictions.sort_values('sort_key').drop('sort_key', axis=1)
            
            # Salvar CSV
            csv_filename = f'predictions_detailed_{dataset_name.lower()}.csv'
            csv_path = self.output_dir / csv_filename
            df_predictions.to_csv(csv_path, index=False)
            
            if self.verbose:
                print(f'\n   💾 CSV detalhado salvo: {csv_filename}')
                
                # Estatísticas por categoria
                category_counts = df_predictions['prediction_category'].value_counts()
                print(f'   📊 Distribuição das predições:')
                for cat in ['TP', 'FP', 'TN', 'FN']:
                    count = category_counts.get(cat, 0)
                    pct = (count / len(df_predictions)) * 100 if len(df_predictions) > 0 else 0
                    print(f'      {cat}: {count:>4} ({pct:>5.1f}%)')
                
        except Exception as e:
            if self.verbose:
                print(f'   ⚠️  Erro ao salvar CSV detalhado: {e}')
                import traceback
                traceback.print_exc()
    
    def plot_evaluation(self, y_true, y_pred, y_proba, dataset_name='Test'):
        """
        Criar visualizações da avaliação do modelo
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            y_proba: Probabilidades preditas
            dataset_name: Nome do conjunto (Test/Validation)
        """
        try:
            # Criar diretório de visualizações
            viz_dir = self.output_dir / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Configurar estilo
            sns.set_style("whitegrid")
            
            # Criar figura com 2 subplots
            fig = plt.figure(figsize=(14, 6))
            
            # ===== SUBPLOT 1: Matriz de Confusão =====
            ax1 = plt.subplot(1, 2, 1)
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['INATIVO (0)', 'ATIVO (1)'],
                       yticklabels=['INATIVO (0)', 'ATIVO (1)'],
                       cbar_kws={'label': 'Contagem'},
                       ax=ax1)
            
            ax1.set_xlabel('Predito', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Verdadeiro', fontsize=12, fontweight='bold')
            ax1.set_title(f'Matriz de Confusão - {dataset_name}', 
                         fontsize=14, fontweight='bold')
            
            # Adicionar porcentagens
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax1.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)',
                            ha='center', va='center', fontsize=10, color='gray')
            
            # ===== SUBPLOT 2: Curva ROC =====
            ax2 = plt.subplot(1, 2, 2)
            
            if len(np.unique(y_true)) == 2:
                from sklearn.metrics import roc_curve, auc
                
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                ax2.plot(fpr, tpr, color='#e74c3c', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax2.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                        label='Random')
                
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('Taxa de Falsos Positivos', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12, fontweight='bold')
                ax2.set_title(f'Curva ROC - {dataset_name}', fontsize=14, fontweight='bold')
                ax2.legend(loc="lower right", fontsize=10)
                ax2.grid(alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Curva ROC disponível apenas\npara classificação binária',
                        ha='center', va='center', fontsize=12,
                        transform=ax2.transAxes)
                ax2.set_title(f'Curva ROC - {dataset_name}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Salvar figura
            viz_file = viz_dir / f'evaluation_{dataset_name.lower()}.png'
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f'   📊 Visualização salva: {viz_file}')
        
        except Exception as e:
            if self.verbose:
                print(f'   ⚠️  Erro ao gerar visualização: {e}')
    
    def save_results(self):
        """Salvar resultados do pipeline"""
        if self.verbose:
            print('💾 Salvando Resultados')
            print('-'*60)
        
        # Adicionar tempo total
        self.stats['end_time'] = datetime.now().isoformat()
        
        # Calcular tempo total
        from datetime import datetime as dt
        if 'start_time' in self.stats:
            start = dt.fromisoformat(self.stats['start_time'])
            end = dt.fromisoformat(self.stats['end_time'])
            self.stats['total_time_seconds'] = (end - start).total_seconds()
        
        # Adicionar resumo
        self.stats['summary'] = {
            'model_type': self.stats.get('model_type', 'Unknown'),
            'best_val_f1': self.stats.get('validation_metrics', {}).get('f1', 0),
            'test_f1': self.stats.get('test_metrics', {}).get('f1', 0),
            'train_samples': self.stats.get('train_samples', 0),
            'val_samples': self.stats.get('val_samples', 0),
            'test_samples': self.stats.get('test_samples', 0)
        }
        
        # Salvar estatísticas
        stats_file = self.output_dir / 'pipeline_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        # Salvar resumo legível
        summary_file = self.output_dir / 'results_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write(' '*25 + 'RESUMO DO PIPELINE DOCKTKINASE\n')
            f.write('='*80 + '\n\n')
            
            f.write(f'Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Dataset: {self.stats.get("dataset", "N/A")}\n')
            f.write(f'Modelo: {self.stats.get("model_type", "N/A")}\n')
            f.write(f'ESM Model: {self.stats.get("esm_model", "N/A")}\n\n')
            
            f.write('CONJUNTOS DE DADOS:\n')
            f.write(f'  • Treino:    {self.stats.get("train_samples", 0):>6} amostras\n')
            f.write(f'  • Validação: {self.stats.get("val_samples", 0):>6} amostras\n')
            f.write(f'  • Teste:     {self.stats.get("test_samples", 0):>6} amostras\n')
            f.write(f'  • Total:     {self.stats.get("total_samples", 0):>6} amostras\n\n')
            
            f.write('DISTRIBUIÇÃO DE CLASSES:\n')
            class_dist = self.stats.get('class_distribution', {})
            for cls, count in class_dist.items():
                pct = count / self.stats.get('total_samples', 1) * 100
                f.write(f'  • Classe {cls}: {count:>6} ({pct:>5.1f}%)\n')
            f.write('\n')
            
            f.write('MÉTRICAS DE PERFORMANCE:\n')
            f.write('='*80 + '\n')
            f.write(f'{"Conjunto":<12} {"F1":>7} {"Acurácia":>9} {"Precisão":>9} {"Recall":>7} {"ROC-AUC":>8}\n')
            f.write('-'*80 + '\n')
            
            # Treino
            train_acc = self.stats.get("train_accuracy", 0)
            f.write(f'{"Treino":<12} {"N/A":>7} {train_acc:>9.4f} {"N/A":>9} {"N/A":>7} {"N/A":>8}\n')
            
            # Validação
            val_metrics = self.stats.get('validation_metrics', {})
            if val_metrics:
                f.write(f'{"Validação":<12} '
                        f'{val_metrics.get("f1", 0):>7.4f} '
                        f'{val_metrics.get("accuracy", 0):>9.4f} '
                        f'{val_metrics.get("precision", 0):>9.4f} '
                        f'{val_metrics.get("recall", 0):>7.4f} '
                        f'{val_metrics.get("roc_auc", 0):>8.4f}\n')
            
            # Teste
            test_metrics = self.stats.get('test_metrics', {})
            if test_metrics:
                f.write(f'{"Teste":<12} '
                        f'{test_metrics.get("f1", 0):>7.4f} '
                        f'{test_metrics.get("accuracy", 0):>9.4f} '
                        f'{test_metrics.get("precision", 0):>9.4f} '
                        f'{test_metrics.get("recall", 0):>7.4f} '
                        f'{test_metrics.get("roc_auc", 0):>8.4f}\n')
            
            f.write('='*80 + '\n\n')
            
            f.write('TEMPOS DE EXECUÇÃO:\n')
            f.write(f'  • Embeddings:   {self.stats.get("embedding_time", 0):>7.2f}s\n')
            f.write(f'  • Treinamento:  {self.stats.get("train_time", 0):>7.2f}s\n')
            f.write(f'  • Avaliação:    {self.stats.get("eval_time", 0):>7.2f}s\n')
            f.write(f'  • Total:        {self.stats.get("total_time_seconds", 0):>7.2f}s\n')
            f.write('='*80 + '\n')
        
        if self.verbose:
            print(f'   📄 Estatísticas JSON: pipeline_stats.json')
            print(f'   📊 Resumo legível:    results_summary.txt')
            print(f'   📁 Diretório:         {self.output_dir}')
            print()
    
    def run(self):
        """Executar pipeline completo"""
        try:
            # 1. Carregar dataset
            df = self.load_dataset()
            
            # 2. Criar labels
            y, df = self.create_labels(df)
            
            # 3. Gerar embeddings
            X = self.generate_embeddings(df, batch_size=8)
            
            # 4. Split estratificado em 3 conjuntos (treino/val/teste) SEM data leaking
            X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = self.stratified_split(X, y)
            
            # 5. Treinar classificador (com validação)
            clf = self.train_classifier(X_train, y_train, X_val, y_val)
            
            # 6. Avaliar no conjunto de validação
            val_metrics = self.evaluate_classifier(clf, X_val, y_val, dataset_name='Validation', 
                                                   df_subset=df, indices=idx_val)
            
            # 7. Avaliar no conjunto de teste (nunca visto antes!)
            test_metrics = self.evaluate_classifier(clf, X_test, y_test, dataset_name='Test',
                                                    df_subset=df, indices=idx_test)
            
            # 8. Salvar resultados
            self.stats['validation_metrics'] = val_metrics
            self.stats['test_metrics'] = test_metrics
            self.save_results()
            
            if self.verbose:
                print('='*80)
                print('✅ PIPELINE CONCLUÍDO COM SUCESSO!')
                print('='*80)
                
                # Tabela de métricas
                print(f'\n{"Conjunto":<12} {"F1":>7} {"Acurácia":>9} {"Precisão":>9} {"Recall":>7} {"ROC-AUC":>8}')
                print('-'*80)
                
                # Treino
                train_m = self.stats.get("train_metrics", {})
                print(f'{"Treino":<12} '
                      f'{train_m.get("f1", 0):>7.4f} '
                      f'{train_m.get("accuracy", 0):>9.4f} '
                      f'{train_m.get("precision", 0):>9.4f} '
                      f'{train_m.get("recall", 0):>7.4f} '
                      f'{train_m.get("roc_auc", 0):>8.4f}')
                
                # Validação
                print(f'{"Validação":<12} '
                      f'{val_metrics["f1"]:>7.4f} '
                      f'{val_metrics["accuracy"]:>9.4f} '
                      f'{val_metrics["precision"]:>9.4f} '
                      f'{val_metrics["recall"]:>7.4f} '
                      f'{val_metrics["roc_auc"]:>8.4f}')
                
                # Teste
                print(f'{"Teste":<12} '
                      f'{test_metrics["f1"]:>7.4f} '
                      f'{test_metrics["accuracy"]:>9.4f} '
                      f'{test_metrics["precision"]:>9.4f} '
                      f'{test_metrics["recall"]:>7.4f} '
                      f'{test_metrics["roc_auc"]:>8.4f}')
                
                print('='*80)
                
                # Resumo adicional
                print(f'\n📈 Resumo de Performance:')
                print(f'   • Melhor Métrica (Val): F1={val_metrics["f1"]:.4f}')
                print(f'   • Generalização (Test): F1={test_metrics["f1"]:.4f}')
                print(f'   • Tempo Total: {time.time() - time.time():.2f}s')
                print(f'   • Amostras Treino/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}')
                print('='*80)
            
            return clf, {'validation': val_metrics, 'test': test_metrics}
            
        except Exception as e:
            print(f'❌ ERRO no pipeline: {e}')
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Pipeline completo DockTKinase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Dataset humano com modelo pequeno (teste rápido - usa Ki/Kd/IC50 automaticamente)
  python run_complete_pipeline.py --dataset human --model esm2_t6_8M_UR50D --max-samples 1000
  
  # Dataset com labels baseados em todas as medidas <= 1000 nM (modo padrão)
  python run_complete_pipeline.py --dataset human --label-method auto
  
  # Dataset com threshold personalizado (500 nM para compostos mais potentes)
  python run_complete_pipeline.py --dataset non_human --label-threshold 500
  
  # Dataset completo com modelo grande (produção)
  python run_complete_pipeline.py --dataset all --model esm2_t36_3B_UR50D --device cuda
  
Métodos de Label (PRIORIDADE: Ki > Kd > IC50 > pchembl):
  - auto: Detecta automaticamente (RECOMENDADO)
          Usa TODAS as medidas disponíveis: Ki, Kd e IC50 combinadas
  - ki: Ki <= threshold (default: 1000 nM) = ATIVO
  - kd: Kd <= threshold (default: 1000 nM) = ATIVO
  - ic50: IC50 <= threshold (default: 1000 nM) = ATIVO
  - pchembl: pchembl_value >= threshold (default: 6.0) = ATIVO
  
Threshold Padrão:
  - Valores de atividade (Ki/Kd/IC50): 1000 nM
    * <= 1000 nM = ATIVO (classe 1) - compostos potentes
    * > 1000 nM = INATIVO (classe 0) - compostos fracos
  
Classes:
  - Classe 0: INATIVO/NEGATIVO (baixa atividade, > 1000 nM)
  - Classe 1: ATIVO/POSITIVO (alta atividade, <= 1000 nM)
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='human',
        choices=['human', 'non_human', 'all'],
        help='Dataset a usar (default: human)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='esm2_t6_8M_UR50D',
        help='Modelo ESM-2 a usar (default: esm2_t6_8M_UR50D)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proporção do conjunto de validação (default: 0.1 = 10%%)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.1,
        help='Proporção do conjunto de teste (default: 0.1 = 10%%)'
    )
    
    parser.add_argument(
        '--label-method',
        type=str,
        default='auto',
        choices=['pchembl', 'ic50', 'ki', 'kd', 'auto'],
        help='Método para criar labels (default: auto)'
    )
    
    parser.add_argument(
        '--label-threshold',
        type=float,
        default=None,
        help='Threshold para labels (default: None = usar padrão do método)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limite de amostras para teste rápido (default: todas)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'auto'],
        help='Device a usar (default: auto)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/pipeline_output',
        help='Diretório para salvar resultados (default: tests/pipeline_output)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso (sem logs)'
    )
    
    args = parser.parse_args()
    
    # Criar e executar pipeline
    pipeline = CompletePipeline(
        dataset_name=args.dataset,
        esm_model=args.model,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        max_samples=args.max_samples,
        device=args.device,
        output_dir=args.output_dir,
        label_method=args.label_method,
        label_threshold=args.label_threshold,
        verbose=not args.quiet
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()
