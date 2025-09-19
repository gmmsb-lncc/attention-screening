"""
Utilitários para manipulação de arquivos.
"""

import os
import glob
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np
import logging

from ..core.exceptions import BuildFileNotFoundError

logger = logging.getLogger(__name__)

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Garante que um diretório existe, criando-o se necessário.
    
    Args:
        directory: Caminho do diretório
        
    Returns:
        Path object do diretório
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def find_files(directory: Union[str, Path], 
               pattern: str = "*", 
               recursive: bool = False) -> List[Path]:
    """
    Encontra arquivos em um diretório usando padrão glob.
    
    Args:
        directory: Diretório para buscar
        pattern: Padrão glob para busca
        recursive: Se deve buscar recursivamente
        
    Returns:
        Lista de caminhos encontrados
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    if recursive:
        glob_pattern = f"**/{pattern}"
        return list(dir_path.glob(glob_pattern))
    else:
        return list(dir_path.glob(pattern))

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Obtém tamanho de arquivo em bytes.
    
    Args:
        file_path: Caminho do arquivo
        
    Returns:
        Tamanho em bytes
    """
    return Path(file_path).stat().st_size

def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Obtém tamanho total de um diretório.
    
    Args:
        directory: Caminho do diretório
        
    Returns:
        Tamanho total em bytes
    """
    total_size = 0
    dir_path = Path(directory)
    
    if dir_path.is_file():
        return get_file_size(dir_path)
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            total_size += get_file_size(file_path)
    
    return total_size

def copy_file(src: Union[str, Path], 
              dst: Union[str, Path], 
              create_dirs: bool = True) -> None:
    """
    Copia arquivo, criando diretórios se necessário.
    
    Args:
        src: Arquivo origem
        dst: Arquivo destino
        create_dirs: Se deve criar diretórios do destino
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise BuildFileNotFoundError(f"Arquivo origem não encontrado: {src}")
    
    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src_path, dst_path)

def move_file(src: Union[str, Path], 
              dst: Union[str, Path], 
              create_dirs: bool = True) -> None:
    """
    Move arquivo, criando diretórios se necessário.
    
    Args:
        src: Arquivo origem
        dst: Arquivo destino
        create_dirs: Se deve criar diretórios do destino
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise BuildFileNotFoundError(f"Arquivo origem não encontrado: {src}")
    
    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.move(str(src_path), str(dst_path))

def delete_file(file_path: Union[str, Path], 
                missing_ok: bool = True) -> None:
    """
    Deleta arquivo.
    
    Args:
        file_path: Caminho do arquivo
        missing_ok: Se deve ignorar arquivo inexistente
    """
    path = Path(file_path)
    
    if not path.exists() and not missing_ok:
        raise BuildFileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    if path.exists():
        path.unlink()

def clean_directory(directory: Union[str, Path], 
                   pattern: str = "*",
                   keep_directory: bool = True) -> None:
    """
    Limpa conteúdo de um diretório.
    
    Args:
        directory: Diretório para limpar
        pattern: Padrão dos arquivos a deletar
        keep_directory: Se deve manter o diretório vazio
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return
    
    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path)
    
    if not keep_directory:
        try:
            dir_path.rmdir()
        except OSError:
            pass  # Diretório não vazio

def read_text_file(file_path: Union[str, Path], 
                  encoding: str = 'utf-8') -> str:
    """
    Lê arquivo de texto.
    
    Args:
        file_path: Caminho do arquivo
        encoding: Encoding do arquivo
        
    Returns:
        Conteúdo do arquivo
    """
    path = Path(file_path)
    if not path.exists():
        raise BuildFileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    return path.read_text(encoding=encoding)

def write_text_file(file_path: Union[str, Path], 
                   content: str,
                   encoding: str = 'utf-8',
                   create_dirs: bool = True) -> None:
    """
    Escreve arquivo de texto.
    
    Args:
        file_path: Caminho do arquivo
        content: Conteúdo a escrever
        encoding: Encoding do arquivo
        create_dirs: Se deve criar diretórios
    """
    path = Path(file_path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    path.write_text(content, encoding=encoding)

def load_tsv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Carrega arquivo TSV como DataFrame.
    
    Args:
        file_path: Caminho do arquivo
        **kwargs: Argumentos adicionais para pd.read_csv
        
    Returns:
        DataFrame com dados do TSV
    """
    path = Path(file_path)
    if not path.exists():
        raise BuildFileNotFoundError(f"Arquivo TSV não encontrado: {file_path}")
    
    return pd.read_csv(path, sep='\t', **kwargs)

def save_tsv(df: pd.DataFrame, 
             file_path: Union[str, Path],
             create_dirs: bool = True,
             **kwargs) -> None:
    """
    Salva DataFrame como TSV.
    
    Args:
        df: DataFrame para salvar
        file_path: Caminho do arquivo
        create_dirs: Se deve criar diretórios
        **kwargs: Argumentos adicionais para df.to_csv
    """
    path = Path(file_path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, sep='\t', index=False, **kwargs)

def load_numpy(file_path: Union[str, Path]) -> np.ndarray:
    """
    Carrega arquivo NumPy.
    
    Args:
        file_path: Caminho do arquivo
        
    Returns:
        Array NumPy
    """
    path = Path(file_path)
    if not path.exists():
        raise BuildFileNotFoundError(f"Arquivo NumPy não encontrado: {file_path}")
    
    return np.load(path)

def save_numpy(array: np.ndarray, 
               file_path: Union[str, Path],
               create_dirs: bool = True) -> None:
    """
    Salva array NumPy.
    
    Args:
        array: Array para salvar
        file_path: Caminho do arquivo
        create_dirs: Se deve criar diretórios
    """
    path = Path(file_path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(path, array)

def get_available_space(directory: Union[str, Path]) -> int:
    """
    Obtém espaço disponível em disco.
    
    Args:
        directory: Diretório para verificar
        
    Returns:
        Espaço disponível em bytes
    """
    path = Path(directory)
    stat = shutil.disk_usage(path)
    return stat.free

def format_size(size_bytes: int) -> str:
    """
    Formata tamanho em bytes para string legível.
    
    Args:
        size_bytes: Tamanho em bytes
        
    Returns:
        String formatada (ex: "1.2 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
