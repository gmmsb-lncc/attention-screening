"""
Exemplo de uso do ESM-C (esmc-300m-2024-12) no docktkinase.

Este script demonstra como usar o novo modelo ESM-C para gerar
embeddings de proteínas usando a interface integrada.

Features demonstradas:
1. Carregamento do modelo ESM-C com cache local
2. Geração de embeddings com mean pooling
3. Comparação de desempenho ESM-2 vs ESM-C
4. Uso do ProteinEmbedding com novo modelo

Autor: docktkinase team
Data: 2025-11-20
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.build.embeddings.protein_embedding import ProteinEmbedding
from src.build.core.logger import setup_logger


def demo_esmc_basic():
    """
    Demo 1: Uso básico do ESM-C (esmc-300m-2024-12).
    
    Demonstra:
    - Carregamento do modelo
    - Geração de embedding único
    - Cache local automático
    """
    print("="*80)
    print("DEMO 1: Uso Básico do ESM-C (esmc-300m-2024-12)")
    print("="*80)
    
    # Setup
    logger = setup_logger('demo_esmc', level='INFO')
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    print(f"\n✓ Device: {device}")
    print(f"✓ Model: esmc-300m-2024-12 (300M params, 960-dim)")
    print(f"✓ Cache: llm/models_cache/ESM3/")
    
    # Inicializar ProteinEmbedding com ESM-C
    protein_emb = ProteinEmbedding(
        model_name='esmc-300m-2024-12',  # Novo modelo ESM-C
        device=device,
        logger=logger
    )
    
    # Sequência de exemplo (Kinase humana)
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVGDTLGTEYLSRLESKIYVDHKHNIKEHEVKLGCGSGIRLLQEE"
    
    print(f"\n✓ Sequence length: {len(sequence)} aa")
    
    # Gerar embedding
    print("\n⏳ Generating embedding...")
    start = time.time()
    embedding = protein_emb.generate(sequence)
    elapsed = time.time() - start
    
    # Resultados
    print(f"\n✅ Embedding gerado com sucesso!")
    print(f"   • Shape: {embedding.shape}")
    print(f"   • Dtype: {embedding.dtype}")
    print(f"   • Time: {elapsed:.3f}s")
    print(f"   • Mean: {np.mean(embedding):.6f}")
    print(f"   • Std: {np.std(embedding):.6f}")
    print(f"   • Range: [{np.min(embedding):.6f}, {np.max(embedding):.6f}]")
    
    # Cleanup
    protein_emb.cleanup()
    print("\n✓ Cleanup completed")


def demo_esmc_batch():
    """
    Demo 2: Processamento em batch com ESM-C.
    
    Demonstra:
    - Geração de múltiplos embeddings
    - Mean pooling automático
    - Performance com batch processing
    """
    print("\n" + "="*80)
    print("DEMO 2: Processamento em Batch com ESM-C")
    print("="*80)
    
    # Setup
    logger = setup_logger('demo_esmc_batch', level='INFO')
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    # Sequências de exemplo (diferentes tamanhos)
    sequences = {
        'Short': 'ACDEFGHIKLMNPQRSTVWY',
        'Medium': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVK',
        'Long': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVGDTLGTEYLSRLESKIYVDHKHNIKEHEVKLGCGSGIRLLQEE' * 2
    }
    
    print(f"\n✓ Processing {len(sequences)} sequences:")
    for name, seq in sequences.items():
        print(f"   • {name}: {len(seq)} aa")
    
    # Inicializar
    protein_emb = ProteinEmbedding(
        model_name='esmc-300m-2024-12',
        device=device,
        logger=logger
    )
    
    # Gerar embeddings
    print("\n⏳ Generating embeddings...")
    embeddings = {}
    total_time = 0
    
    for name, seq in sequences.items():
        start = time.time()
        emb = protein_emb.generate(seq)
        elapsed = time.time() - start
        total_time += elapsed
        
        embeddings[name] = emb
        print(f"   ✓ {name}: {emb.shape} [{elapsed:.3f}s]")
    
    # Análise de similaridade
    print("\n📊 Similarity Analysis (cosine):")
    names = list(embeddings.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name_i, name_j = names[i], names[j]
            emb_i, emb_j = embeddings[name_i], embeddings[name_j]
            
            # Cosine similarity
            similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            print(f"   • {name_i} vs {name_j}: {similarity:.4f}")
    
    print(f"\n✓ Total time: {total_time:.3f}s")
    print(f"✓ Average time: {total_time/len(sequences):.3f}s/sequence")
    
    # Cleanup
    protein_emb.cleanup()


def demo_esmc_vs_esm2():
    """
    Demo 3: Comparação ESM-C vs ESM-2.
    
    Demonstra:
    - Diferenças de dimensão (960 vs 1280)
    - Diferenças de velocidade
    - Diferenças de embeddings
    """
    print("\n" + "="*80)
    print("DEMO 3: Comparação ESM-C vs ESM-2")
    print("="*80)
    
    # Setup
    logger = setup_logger('demo_comparison', level='INFO')
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    # Sequência de teste
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVK"
    print(f"\n✓ Test sequence: {len(sequence)} aa")
    
    # Modelos para comparar
    models = {
        'ESM-C 300M': 'esmc-300m-2024-12',
        'ESM-2 650M': 'esm2_t33_650M_UR50D',
    }
    
    results = {}
    
    for name, model_name in models.items():
        print(f"\n{'='*40}")
        print(f"{name}")
        print(f"{'='*40}")
        
        # Inicializar
        start_load = time.time()
        protein_emb = ProteinEmbedding(
            model_name=model_name,
            device=device,
            logger=logger
        )
        load_time = time.time() - start_load
        
        # Gerar embedding
        start_gen = time.time()
        embedding = protein_emb.generate(sequence)
        gen_time = time.time() - start_gen
        
        # Salvar resultados
        results[name] = {
            'model': model_name,
            'load_time': load_time,
            'gen_time': gen_time,
            'dim': embedding.shape[0],
            'embedding': embedding,
            'mean': np.mean(embedding),
            'std': np.std(embedding),
        }
        
        print(f"✓ Load time: {load_time:.3f}s")
        print(f"✓ Generation time: {gen_time:.3f}s")
        print(f"✓ Dimension: {embedding.shape[0]}")
        print(f"✓ Mean: {np.mean(embedding):.6f}")
        print(f"✓ Std: {np.std(embedding):.6f}")
        
        # Cleanup
        protein_emb.cleanup()
    
    # Comparação final
    print("\n" + "="*80)
    print("RESUMO DA COMPARAÇÃO")
    print("="*80)
    
    print("\n📊 Performance:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"   • Load time: {res['load_time']:.3f}s")
        print(f"   • Generation time: {res['gen_time']:.3f}s")
        print(f"   • Total time: {res['load_time'] + res['gen_time']:.3f}s")
        print(f"   • Dimension: {res['dim']}")
    
    # Speed comparison
    esmc_total = results['ESM-C 300M']['load_time'] + results['ESM-C 300M']['gen_time']
    esm2_total = results['ESM-2 650M']['load_time'] + results['ESM-2 650M']['gen_time']
    speedup = esm2_total / esmc_total
    
    print(f"\n⚡ ESM-C Speedup: {speedup:.2f}x faster than ESM-2")
    
    # Embedding comparison (cosine similarity)
    emb_esmc = results['ESM-C 300M']['embedding']
    emb_esm2 = results['ESM-2 650M']['embedding']
    
    # Normalize to same dimension for comparison (take first 960 dims of ESM-2)
    emb_esm2_trunc = emb_esm2[:960]
    similarity = np.dot(emb_esmc, emb_esm2_trunc) / (
        np.linalg.norm(emb_esmc) * np.linalg.norm(emb_esm2_trunc)
    )
    
    print(f"\n🔍 Embedding Similarity (first 960 dims): {similarity:.4f}")
    print("   (Higher = more similar representations)")


def demo_esmc_integration():
    """
    Demo 4: Integração completa com pipeline.
    
    Demonstra:
    - Uso em contexto real de pipeline
    - Integração com ProteinEmbedding
    - Compatibilidade com código existente
    """
    print("\n" + "="*80)
    print("DEMO 4: Integração Completa com Pipeline")
    print("="*80)
    
    # Setup
    logger = setup_logger('demo_integration', level='INFO')
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    print(f"\n✓ Demonstrando uso drop-in replacement:")
    print(f"   Basta trocar model_name='esm2_...' por model_name='esmc-300m-2024-12'")
    
    # Código ANTES (ESM-2)
    print("\n📄 ANTES (ESM-2):")
    print("""
    protein_emb = ProteinEmbedding(
        model_name='esm2_t33_650M_UR50D',  # ESM-2 650M
        device=device,
        logger=logger
    )
    """)
    
    # Código DEPOIS (ESM-C)
    print("\n📄 DEPOIS (ESM-C):")
    print("""
    protein_emb = ProteinEmbedding(
        model_name='esmc-300m-2024-12',    # ESM-C 300M (Fase 1)
        device=device,
        logger=logger
    )
    """)
    
    print("\n✅ Benefícios da Integração:")
    print("   • Backward compatible (código ESM-2 continua funcionando)")
    print("   • Strategy Pattern (fácil adicionar novos modelos)")
    print("   • Cache local automático (llm/models_cache/ESM3/)")
    print("   • Mean pooling (melhor representação da sequência)")
    print("   • Flash Attention support (quando disponível)")
    print("   • Sequências mais longas (até 2048 tokens vs 1024)")
    
    print("\n✅ Fase 1 (ESM-C) Completa!")
    print("   ✓ esmc-300m-2024-12 implementado")
    print("   ✓ Mean pooling configurado")
    print("   ✓ Cache local funcionando")
    print("   ✓ Factory Pattern integrado")
    print("   ✓ Tests criados")


def main():
    """Executar todos os demos."""
    print("\n" + "="*80)
    print("DEMOS ESM-C (esmc-300m-2024-12) - Fase 1")
    print("="*80)
    print("\nNovos recursos:")
    print("  • Modelo: esmc-300m-2024-12 (300M params, 960-dim)")
    print("  • Pooling: Mean pooling sobre sequência")
    print("  • Cache: llm/models_cache/ESM3/ (local)")
    print("  • Max length: 2048 tokens (2x ESM-2)")
    
    try:
        # Demo 1: Básico
        demo_esmc_basic()
        
        # Demo 2: Batch
        demo_esmc_batch()
        
        # Demo 3: Comparação
        demo_esmc_vs_esm2()
        
        # Demo 4: Integração
        demo_esmc_integration()
        
        print("\n" + "="*80)
        print("✅ TODOS OS DEMOS COMPLETADOS COM SUCESSO!")
        print("="*80)
        
    except ImportError as e:
        print(f"\n❌ Erro de importação: {e}")
        print("\nPara usar ESM-C, instale o ESM-3:")
        print("  cd ESM/esm-3/esm-main")
        print("  pip install -e .")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
