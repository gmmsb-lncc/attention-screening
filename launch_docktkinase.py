#!/usr/bin/env python3
"""
Launcher para DockTKinase - Configura ambiente e inicia sistema.
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Configura ambiente Python."""
    # Adicionar src ao path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("🚀 DockTKinase Launcher")
    print("=" * 30)
    print(f"📁 Projeto: {Path.cwd()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📦 Src path: {src_path}")

def test_system():
    """Testa se o sistema está funcionando."""
    try:
        from classifier.modular_classifier import main as classifier_main
        from classifier.modular_pipeline import MLPEmbeddingPipeline
        from classifier.models.mlp_classifier import MLPEmbeddingClassifier
        from classifier.utils.import_utils import safe_import_optional
        
        print("✅ Sistema modularizado carregado com sucesso!")
        
        # Note: Pipeline precisa de argumentos, então apenas validamos que a classe existe
        print(f"✅ Pipeline: {MLPEmbeddingPipeline.__name__} disponível")
        
        # Testar modelo
        model = MLPEmbeddingClassifier(input_dim=100, hidden_dim=64, dropout=0.3)
        print("✅ Modelo MLP: OK")
        
        # Verificar dependências opcionais
        optuna_available = safe_import_optional("optuna", "otimização")
        pyspark_available = safe_import_optional("pyspark", "processamento distribuído")
        
        print(f"� Optuna: {'✅ Disponível' if optuna_available else '⚠️  Não disponível'}")
        print(f"🔧 PySpark: {'✅ Disponível' if pyspark_available else '⚠️  Não disponível'}")
        
        print("")
        print("Sistema pronto para uso!")
        print("Para começar:")
        print("  from classifier.modular_pipeline import MLPEmbeddingPipeline")
        print("  pipeline = MLPEmbeddingPipeline(embeddings_path, labels_path)")
        print("  # ou usar CLI:")
        print("  python src/classifier/modular_classifier.py --help")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar sistema: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    setup_environment()
    if test_system():
        print("\n🎉 DockTKinase está pronto!")
    else:
        print("\n⚠️  Verifique a instalação")
        sys.exit(1)
