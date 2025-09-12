#!/usr/bin/env python3
"""
Script wrapper para executar o DockTKinase Classifier com ambiente configurado.
"""

import sys
import os
from pathlib import Path

# Configuração do path
project_root = Path(__file__).parent.parent.absolute()
classifier_path = project_root / "src" / "classifier"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(classifier_path))

# Muda para o diretório do classifier
os.chdir(classifier_path)

# Importa e executa o main
if __name__ == "__main__":
    try:
        import src.classifier.main
        # Executa o main com os argumentos da linha de comando
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"Erro ao executar classifier: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
