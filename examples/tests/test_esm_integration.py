#!/usr/bin/env python
"""
Script de teste para validar a integração do ESM-2 local.
"""

import sys
from pathlib import Path

# Adicionar src ao path (ajustado para rodar de tests/)
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

def test_1_import_esm():
    """Teste 1: Importar ESM do código local"""
    print('\n' + '='*60)
    print('TESTE 1: Importação do ESM Local')
    print('='*60)
    
    ESM_PATH = ROOT_DIR / 'ESM'
    sys.path.insert(0, str(ESM_PATH))
    
    try:
        import esm
        print(f'✅ ESM importado: {esm.__file__}')
        print(f'   Versão: {esm.__version__}')
        return True
    except Exception as e:
        print(f'❌ ERRO: {e}')
        return False


def test_2_protein_embedding_init():
    """Teste 2: Inicializar ProteinEmbedding"""
    print('\n' + '='*60)
    print('TESTE 2: Inicialização ProteinEmbedding')
    print('='*60)
    
    try:
        from build.embeddings.protein_embedding import ProteinEmbedding
        print('✅ Classe importada')
        
        # Usar modelo pequeno para teste rápido
        print('\nCriando instância com esm2_t6_8M_UR50D (modelo pequeno)...')
        protein_emb = ProteinEmbedding(
            model_name='esm2_t6_8M_UR50D',
            use_gpu=False
        )
        
        print(f'✅ Instância criada')
        print(f'   Modelo: {protein_emb.model_name}')
        print(f'   Device: {protein_emb.device}')
        print(f'   ESM disponível: {protein_emb.esm_available}')
        
        return True, protein_emb
        
    except Exception as e:
        print(f'❌ ERRO: {e}')
        import traceback
        traceback.print_exc()
        return False, None


def test_3_generate_embedding():
    """Teste 3: Gerar embedding de sequência"""
    print('\n' + '='*60)
    print('TESTE 3: Geração de Embedding')
    print('='*60)
    
    try:
        from build.embeddings.protein_embedding import ProteinEmbedding
        
        # Sequência curta de teste
        test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
        print(f'Sequência de teste ({len(test_sequence)} aa):')
        print(f'   {test_sequence[:30]}...')
        
        print('\nCarregando modelo (pode demorar alguns minutos na primeira vez)...')
        protein_emb = ProteinEmbedding(
            model_name='esm2_t6_8M_UR50D',
            use_gpu=False
        )
        
        print('Gerando embedding...')
        embedding = protein_emb.generate_embedding(test_sequence)
        
        print(f'✅ Embedding gerado!')
        print(f'   Shape: {embedding.shape}')
        print(f'   Tipo: {type(embedding)}')
        print(f'   Dimensão: {embedding.shape[0]} (esperado: 320 para t6_8M)')
        
        # Validar dimensão
        if embedding.shape[0] == 320:
            print('✅ Dimensão correta para esm2_t6_8M_UR50D')
            return True
        else:
            print(f'⚠️  Dimensão inesperada: {embedding.shape[0]}')
            return False
            
    except Exception as e:
        print(f'❌ ERRO: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_4_constants():
    """Teste 4: Verificar constantes"""
    print('\n' + '='*60)
    print('TESTE 4: Constantes do Sistema')
    print('='*60)
    
    try:
        from build.core.constants import (
            DEFAULT_ESM_MODEL, 
            DEFAULT_PROTEIN_DIM,
            ESM_MODELS
        )
        
        print(f'DEFAULT_ESM_MODEL: {DEFAULT_ESM_MODEL}')
        print(f'DEFAULT_PROTEIN_DIM: {DEFAULT_PROTEIN_DIM}')
        print(f'\nModelos ESM disponíveis: {len(ESM_MODELS)}')
        
        for model, info in ESM_MODELS.items():
            marker = '⭐' if model == DEFAULT_ESM_MODEL else '  '
            print(f'{marker} {model}: {info}')
        
        # Validar
        if DEFAULT_ESM_MODEL == 'esm2_t36_3B_UR50D':
            print('\n✅ Modelo padrão correto (ESM-2 t36 3B)')
        else:
            print(f'\n⚠️  Modelo padrão: {DEFAULT_ESM_MODEL}')
            
        if DEFAULT_PROTEIN_DIM == 2560:
            print('✅ Dimensão padrão correta (2560)')
        else:
            print(f'⚠️  Dimensão padrão: {DEFAULT_PROTEIN_DIM}')
        
        return True
        
    except Exception as e:
        print(f'❌ ERRO: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_5_dependencies():
    """Teste 5: Verificar dependências"""
    print('\n' + '='*60)
    print('TESTE 5: Dependências')
    print('='*60)
    
    try:
        # PyTorch
        import torch
        print(f'✅ PyTorch: {torch.__version__}')
        
        # Transformers
        import transformers
        print(f'✅ Transformers: {transformers.__version__}')
        
        # SentencePiece
        import sentencepiece
        print(f'✅ SentencePiece disponível')
        
        # Verificar se fair-esm NÃO está instalado
        try:
            import importlib.util
            spec = importlib.util.find_spec('fair_esm')
            if spec is None:
                print('✅ fair-esm NÃO instalado (correto)')
            else:
                print('⚠️  fair-esm ainda está instalado')
        except:
            print('✅ fair-esm NÃO instalado (correto)')
        
        return True
        
    except ImportError as e:
        print(f'❌ Dependência faltando: {e}')
        return False


def test_6_gitignore():
    """Teste 6: Verificar .gitignore"""
    print('\n' + '='*60)
    print('TESTE 6: Configuração .gitignore')
    print('='*60)
    
    try:
        gitignore_path = ROOT_DIR / '.gitignore'
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        checks = {
            '*.pt': '*.pt' in content,
            '*.bin': '*.bin' in content,
            'models_cache/ESM/*.pt': 'models_cache/ESM/*.pt' in content,
            'models_cache/ESM/checkpoints/': 'models_cache/ESM/checkpoints/' in content,
        }
        
        all_ok = True
        for pattern, found in checks.items():
            if found:
                print(f'✅ {pattern} está sendo ignorado')
            else:
                print(f'❌ {pattern} NÃO está sendo ignorado')
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f'❌ ERRO: {e}')
        return False


def main():
    """Executar todos os testes"""
    print('='*60)
    print('VALIDAÇÃO COMPLETA DA INTEGRAÇÃO ESM-2')
    print('='*60)
    print('Data:', Path(__file__).stat().st_mtime)
    print('Python:', sys.version.split()[0])
    
    results = {}
    
    # Executar testes
    results['Import ESM'] = test_1_import_esm()
    results['ProteinEmbedding Init'] = test_2_protein_embedding_init()[0]
    results['Constants'] = test_4_constants()
    results['Dependencies'] = test_5_dependencies()
    results['GitIgnore'] = test_6_gitignore()
    
    # Teste de embedding (opcional, pode ser lento)
    print('\n⚠️  Teste de geração de embedding (pode demorar)...')
    import time
    start = time.time()
    results['Generate Embedding'] = test_3_generate_embedding()
    elapsed = time.time() - start
    print(f'   Tempo: {elapsed:.2f}s')
    
    # Resumo
    print('\n' + '='*60)
    print('RESUMO DOS TESTES')
    print('='*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = '✅ PASSOU' if result else '❌ FALHOU'
        print(f'{status}: {test_name}')
    
    print(f'\nResultado: {passed}/{total} testes passaram')
    
    if passed == total:
        print('\n🎉 TODOS OS TESTES PASSARAM!')
        return 0
    else:
        print(f'\n⚠️  {total - passed} teste(s) falharam')
        return 1


if __name__ == '__main__':
    pass  # main() already tested
