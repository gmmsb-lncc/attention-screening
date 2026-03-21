#!/usr/bin/env python3
"""
Script avançado para corrigir TODOS os imports relativos para absolutos.
"""

import os
import re
from pathlib import Path

def fix_all_imports(file_path: Path):
    """Corrige TODOS os tipos de imports relativos."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Substituições mais agressivas e específicas
    replacements = [
        # Imports com .. (subindo níveis)
        (r'from \.\.core', 'from build.core'),
        (r'from \.\.pipeline', 'from build.pipeline'),  
        (r'from \.\.matrix', 'from build.matrix'),
        (r'from \.\.embeddings', 'from build.embeddings'),
        (r'from \.\.labels', 'from build.labels'),
        (r'from \.\.validation', 'from build.validation'),
        (r'from \.\.utils', 'from build.utils'),
        
        # Imports com . (mesmo nível) 
        (r'from \.core', 'from build.core'),
        (r'from \.pipeline', 'from build.pipeline'),
        (r'from \.matrix', 'from build.matrix'),
        (r'from \.embeddings', 'from build.embeddings'),
        (r'from \.labels', 'from build.labels'),
        (r'from \.validation', 'from build.validation'),
        (r'from \.utils', 'from build.utils'),
        
        # Imports específicos dentro de cada módulo
        (r'from \.constants', 'from build.core.constants'),
        (r'from \.exceptions', 'from build.core.exceptions'),
        (r'from \.config', 'from build.core.config'),
        (r'from \.base_builder', 'from build.core.base_builder'),
        
        (r'from \.build_pipeline', 'from build.pipeline.build_pipeline'),
        
        (r'from \.base_embedding', 'from build.embeddings.base_embedding'),
        (r'from \.protein_embedding', 'from build.embeddings.protein_embedding'),
        (r'from \.ligand_embedding', 'from build.embeddings.ligand_embedding'),
        
        (r'from \.base_matrix', 'from build.matrix.base_matrix'),
        (r'from \.embedding_matrix', 'from build.matrix.embedding_matrix'),
        (r'from \.kinase_matrix', 'from build.matrix.kinase_matrix'),
        
        (r'from \.base_labels', 'from build.labels.base_labels'),
        (r'from \.binary_labels', 'from build.labels.binary_labels'),
        (r'from \.interaction_labels', 'from build.labels.interaction_labels'),
        
        (r'from \.base_validator', 'from build.validation.base_validator'),
        (r'from \.matrix_validator', 'from build.validation.matrix_validator'),
        
        (r'from \.memory_utils', 'from build.utils.memory_utils'),
        (r'from \.file_utils', 'from build.utils.file_utils'),
        (r'from \.spark_utils', 'from build.utils.spark_utils'),
        (r'from \.logging_utils', 'from build.utils.logging_utils'),
        
        # Imports ambíguos que podem existir
        (r'\bfrom core\b(?!\.)', 'from build.core'),
        (r'\bfrom embeddings\b(?!\.)', 'from build.embeddings'),
        (r'\bfrom matrix\b(?!\.)', 'from build.matrix'),
        (r'\bfrom labels\b(?!\.)', 'from build.labels'),
        (r'\bfrom validation\b(?!\.)', 'from build.validation'),
        (r'\bfrom utils\b(?!\.)', 'from build.utils'),
        
        # Imports específicos por submodule
        (r'from core\.constants', 'from build.core.constants'),
        (r'from core\.exceptions', 'from build.core.exceptions'),
        (r'from core\.config', 'from build.core.config'),
        (r'from core\.base_builder', 'from build.core.base_builder'),
        
        # Import statements que referenciam módulos sem prefixo build
        (r'import core\.', 'import build.core.'),
        (r'import embeddings\.', 'import build.embeddings.'),
        (r'import matrix\.', 'import build.matrix.'),
        (r'import labels\.', 'import build.labels.'),
        (r'import validation\.', 'import build.validation.'),
        (r'import utils\.', 'import build.utils.'),
    ]
    
    # Aplicar as substituições
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Salvar se houve mudanças
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed: {file_path}")
        return True
    return False

def main():
    """Função principal."""
    src_build_dir = Path("src/build")
    
    if not src_build_dir.exists():
        print("❌ Directory src/build not found!")
        return
    
    print("🔧 Fixing ALL imports in Python files...")
    
    files_fixed = 0
    
    # Processar todos os arquivos .py
    for py_file in src_build_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        if fix_all_imports(py_file):
            files_fixed += 1
    
    print(f"\n✅ Fixed imports in {files_fixed} files")

if __name__ == "__main__":
    main()
