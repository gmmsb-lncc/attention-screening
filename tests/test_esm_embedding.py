#!/usr/bin/env python
"""Teste de geração de embedding com ESM-2 local"""

import sys
import time
from pathlib import Path
import numpy as np

# Adicionar paths (ajustado para rodar de tests/)
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))
ESM_PATH = ROOT_DIR / 'ESM'
sys.path.insert(0, str(ESM_PATH))

print('='*60)
print('TESTE: Geração de Embedding com ESM-2 Local')
print('='*60)

# Sequência de teste (Kinase humana real)
test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

print(f'\n📝 Sequência de teste:')
print(f'   Tipo: Proteína kinase humana')
print(f'   Tamanho: {len(test_sequence)} aminoácidos')
print(f'   Primeiros 50 aa: {test_sequence[:50]}...')

print(f'\n🔧 Configuração:')
print(f'   Modelo: esm2_t6_8M_UR50D (modelo pequeno para teste rápido)')
print(f'   Device: CPU')
print(f'   Dimensão esperada: 320')

print(f'\n⏳ Carregando modelo ESM-2...')
start_load = time.time()

try:
    # Import direto do módulo de embedding
    from build.core.exceptions import DependencyError, EmbeddingError, ModelLoadError
    
    # Import ESM
    import esm
    import torch
    
    print(f'   ✅ ESM importado de: {Path(esm.__file__).parent}')
    print(f'   ✅ PyTorch {torch.__version__}')
    
    # Carregar modelo pequeno
    model_name = 'esm2_t6_8M_UR50D'
    print(f'\n   Baixando/carregando {model_name}...')
    print(f'   (primeira vez: ~30 MB download)')
    
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    load_time = time.time() - start_load
    print(f'   ✅ Modelo carregado em {load_time:.2f}s')
    
    # Preparar dados
    print(f'\n🔄 Processando sequência...')
    data = [
        ("test_protein", test_sequence),
    ]
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    print(f'   ✅ Batch preparado')
    print(f'   Tokens shape: {batch_tokens.shape}')
    
    # Gerar embedding
    print(f'\n⚡ Gerando embedding...')
    start_gen = time.time()
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers])
    
    # Extrair CLS token (representação da sequência)
    embedding = results["representations"][model.num_layers][0, 0]  # [0, 0] = primeiro item, CLS token
    embedding_np = embedding.cpu().numpy()
    
    gen_time = time.time() - start_gen
    print(f'   ✅ Embedding gerado em {gen_time:.2f}s')
    
    # Validar resultado
    print(f'\n📊 Resultados:')
    print(f'   Shape: {embedding_np.shape}')
    print(f'   Dtype: {embedding_np.dtype}')
    print(f'   Dimensão: {embedding_np.shape[0]}')
    print(f'   Min value: {embedding_np.min():.4f}')
    print(f'   Max value: {embedding_np.max():.4f}')
    print(f'   Mean: {embedding_np.mean():.4f}')
    print(f'   Std: {embedding_np.std():.4f}')
    
    # Primeiros 10 valores
    print(f'\n   Primeiros 10 valores:')
    print(f'   {embedding_np[:10]}')
    
    # Validações
    print(f'\n✅ Validações:')
    
    checks = []
    
    # Dimensão correta
    if embedding_np.shape[0] == 320:
        print(f'   ✅ Dimensão correta (320 para esm2_t6_8M_UR50D)')
        checks.append(True)
    else:
        print(f'   ❌ Dimensão incorreta: {embedding_np.shape[0]} (esperado: 320)')
        checks.append(False)
    
    # Sem NaN
    if not np.isnan(embedding_np).any():
        print(f'   ✅ Sem valores NaN')
        checks.append(True)
    else:
        print(f'   ❌ Contém valores NaN')
        checks.append(False)
    
    # Sem Inf
    if not np.isinf(embedding_np).any():
        print(f'   ✅ Sem valores Inf')
        checks.append(True)
    else:
        print(f'   ❌ Contém valores Inf')
        checks.append(False)
    
    # Valores razoáveis
    if -10 < embedding_np.mean() < 10:
        print(f'   ✅ Valores em range razoável')
        checks.append(True)
    else:
        print(f'   ⚠️  Valores fora do range esperado')
        checks.append(False)
    
    # Resumo final
    print(f'\n' + '='*60)
    if all(checks):
        print('✅ TESTE DE GERAÇÃO DE EMBEDDING: SUCESSO!')
        print('='*60)
        print(f'\nTempo total: {load_time + gen_time:.2f}s')
        print(f'  • Carregamento do modelo: {load_time:.2f}s')
        print(f'  • Geração do embedding: {gen_time:.2f}s')
        print(f'\n🎉 ESM-2 local está funcionando perfeitamente!')
        print(f'\nPróximo passo:')
        print(f'  • Testar com modelo maior (esm2_t36_3B_UR50D)')
        print(f'  • Integrar com pipeline completo')
        pass  # Success
    else:
        print('⚠️  TESTE PASSOU COM RESSALVAS')
        print('='*60)
        print(f'Algumas validações falharam. Revisar resultados.')
        raise AssertionError("Test failed")
    
except Exception as e:
    print(f'\n❌ ERRO durante geração de embedding:')
    print(f'   {e}')
    import traceback
    traceback.print_exc()
    raise AssertionError("Test failed")
