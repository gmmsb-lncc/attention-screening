"""
Módulo responsável por carregar e validar dados de resultados.

Princípios aplicados:
- Single Responsibility: apenas carrega dados
- KISS: lógica simples e direta
- Clean Code: funções pequenas e bem nomeadas
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class ResultsLoader:
    """Carrega resultados de arquivos JSON."""
    
    def __init__(self, files: List[str]):
        """
        Inicializa o carregador de resultados.
        
        Args:
            files: Lista de caminhos para arquivos JSON
        """
        self.files = [Path(f) for f in files]
    
    def load_all(self) -> Dict[str, Dict]:
        """
        Carrega todos os arquivos de resultados.
        
        Returns:
            Dict com nome do modelo -> dados completos
        """
        results = {}
        
        for file_path in self.files:
            result = self._load_single_file(file_path)
            if result:
                model_name, data = result
                results[model_name] = data
        
        return results
    
    def _load_single_file(self, file_path: Path) -> Optional[tuple]:
        """
        Carrega um único arquivo JSON.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Tupla (nome_modelo, dados) ou None se falhar
        """
        if not file_path.exists():
            print(f"⚠️  Arquivo não encontrado: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            model_name = self._extract_model_name(data, file_path)
            return (model_name, data)
            
        except json.JSONDecodeError as e:
            print(f"❌ Erro ao decodificar JSON {file_path}: {e}")
            return None
        except Exception as e:
            print(f"❌ Erro ao carregar {file_path}: {e}")
            return None
    
    @staticmethod
    def _extract_model_name(data: Dict, file_path: Path) -> str:
        """
        Extrai o nome do modelo dos dados ou do nome do arquivo.
        
        Args:
            data: Dados carregados do JSON
            file_path: Caminho do arquivo
            
        Returns:
            Nome do modelo
        """
        # Tentar extrair da configuração
        if 'config' in data:
            model_name = data['config'].get('esm_model')
            if model_name:
                return model_name
        
        # Fallback: extrair do nome do arquivo
        # Ex: "integrated_results_esm2_t36_3B.json" -> "esm2_t36_3B"
        name = file_path.stem
        if 'integrated_results_' in name:
            return name.replace('integrated_results_', '')
        
        return name


def load_results_from_files(files: List[str]) -> Dict[str, Dict]:
    """
    Função utilitária para carregar resultados.
    
    Args:
        files: Lista de caminhos para arquivos JSON
        
    Returns:
        Dict com nome do modelo -> dados completos
    """
    loader = ResultsLoader(files)
    return loader.load_all()
