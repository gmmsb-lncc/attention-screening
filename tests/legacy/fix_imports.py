#!/usr/bin/env python3
"""
Script para corrigir imports relativos para absolutos em todo o sistema build.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path: Path):
    """Corrige imports relativos para absolutos em um arquivo."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Mapear padrões de imports relativos para absolutos
    replacements = [
        # Imports do core
        (r'from \.core import', 'from build.core import'),
        (r'from \.core\.', 'from build.core.'),
        
        # Imports do pipeline
        (r'from \.pipeline import', 'from build.pipeline import'),
        (r'from \.pipeline\.', 'from build.pipeline.'),
        
        # Imports do matrix
        (r'from \.matrix import', 'from build.matrix import'),
        (r'from \.matrix\.', 'from build.matrix.'),
        
        # Imports do embeddings
        (r'from \.embeddings import', 'from build.embeddings import'),
        (r'from \.embeddings\.', 'from build.embeddings.'),
        
        # Imports do labels
        (r'from \.labels import', 'from build.labels import'),
        (r'from \.labels\.', 'from build.labels.'),
        
        # Imports do validation
        (r'from \.validation import', 'from build.validation import'),
        (r'from \.validation\.', 'from build.validation.'),
        
        # Imports do utils
        (r'from \.utils import', 'from build.utils import'),
        (r'from \.utils\.', 'from build.utils.'),
        
        # Imports relativos específicos dentro dos submodules
        (r'from \.constants import', 'from build.core.constants import'),
        (r'from \.exceptions import', 'from build.core.exceptions import'),
        (r'from \.config import', 'from build.core.config import'),
        (r'from \.base_builder import', 'from build.core.base_builder import'),
        
        (r'from \.build_pipeline import', 'from build.pipeline.build_pipeline import'),
        
        (r'from \.base_embedding import', 'from build.embeddings.base_embedding import'),
        (r'from \.protein_embedding import', 'from build.embeddings.protein_embedding import'),
        (r'from \.ligand_embedding import', 'from build.embeddings.ligand_embedding import'),
        
        (r'from \.base_matrix import', 'from build.matrix.base_matrix import'),
        (r'from \.embedding_matrix import', 'from build.matrix.embedding_matrix import'),
        (r'from \.kinase_matrix import', 'from build.matrix.kinase_matrix import'),
        
        (r'from \.base_labels import', 'from build.labels.base_labels import'),
        (r'from \.binary_labels import', 'from build.labels.binary_labels import'),
        (r'from \.interaction_labels import', 'from build.labels.interaction_labels import'),
        
        (r'from \.base_validator import', 'from build.validation.base_validator import'),
        (r'from \.matrix_validator import', 'from build.validation.matrix_validator import'),
        
        (r'from \.memory_utils import', 'from build.utils.memory_utils import'),
        (r'from \.file_utils import', 'from build.utils.file_utils import'),
        (r'from \.spark_utils import', 'from build.utils.spark_utils import'),
        (r'from \.logging_utils import', 'from build.utils.logging_utils import'),
        
        # Imports específicos que podem aparecer
        (r'from core import', 'from build.core import'),
        (r'from core\.', 'from build.core.'),
        (r'from embeddings import', 'from build.embeddings import'),
        (r'from matrix import', 'from build.matrix import'),
        (r'from labels import', 'from build.labels import'),
        (r'from validation import', 'from build.validation import'),
        (r'from utils import', 'from build.utils import'),
    ]
    
    # Aplicar as substituições
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Salvar apenas se houve mudanças
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed imports in: {file_path}")
        return True
    return False

def main():
    """Função principal."""
    src_build_dir = Path("src/build")
    
    if not src_build_dir.exists():
        print("❌ Directory src/build not found!")
        return
    
    print("🔧 Fixing imports in all Python files...")
    
    files_fixed = 0
    total_files = 0
    
    # Processar todos os arquivos .py
    for py_file in src_build_dir.rglob("*.py"):
        if py_file.name == "__pycache__":
            continue
            
        total_files += 1
        if fix_imports_in_file(py_file):
            files_fixed += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files processed: {total_files}")
    print(f"   Files with fixes: {files_fixed}")
    print(f"   Files unchanged: {total_files - files_fixed}")
    
    if files_fixed > 0:
        print(f"\n✅ Import fixes completed! Run the review again to verify.")
    else:
        print(f"\n📝 No changes needed.")

if __name__ == "__main__":
    main()
