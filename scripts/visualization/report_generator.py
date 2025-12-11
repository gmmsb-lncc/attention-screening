"""
Módulo para geração de relatórios em formato markdown.

Princípios aplicados:
- Single Responsibility: apenas gera relatórios
- DRY: funções reutilizáveis para formatação
- Clean Code: separação clara entre formatação e conteúdo
"""

from pathlib import Path
from typing import List, Dict
from datetime import datetime


class ReportGenerator:
    """Gera relatórios markdown com resultados de comparação."""
    
    def __init__(self, output_dir: Path):
        """
        Inicializa o gerador de relatórios.
        
        Args:
            output_dir: Diretório para salvar o relatório
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary(self, 
                        classification_data: List[Dict],
                        regression_data: List[Dict],
                        embedding_data: List[Dict],
                        overall_data: List[Dict]) -> Path:
        """
        Gera relatório completo de comparação.
        
        Args:
            classification_data: Métricas de classificação
            regression_data: Métricas de regressão
            embedding_data: Informações de embeddings
            overall_data: Scores gerais
            
        Returns:
            Caminho do arquivo salvo
        """
        lines = []
        
        # Cabeçalho
        lines.extend(self._create_header())
        
        # Ranking geral
        lines.extend(self._create_ranking_section(overall_data))
        
        # Classificação
        lines.extend(self._create_classification_section(classification_data))
        
        # Regressão
        lines.extend(self._create_regression_section(regression_data))
        
        # Embeddings
        lines.extend(self._create_embedding_section(embedding_data))
        
        # Conclusões
        lines.extend(self._create_conclusions(overall_data, classification_data, 
                                              regression_data))
        
        # Salvar
        output_path = self.output_dir / 'SUMMARY.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def _create_header(self) -> List[str]:
        """Cria cabeçalho do relatório."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return [
            '# Relatório de Comparação de Modelos de Proteína',
            '',
            f'**Data de geração:** {timestamp}',
            '',
            '---',
            ''
        ]
    
    def _create_ranking_section(self, data: List[Dict]) -> List[str]:
        """Cria seção de ranking geral."""
        sorted_data = sorted(data, key=lambda x: x['Overall_Score'], reverse=True)
        
        lines = [
            '## 🏆 Ranking Geral',
            '',
            '| Posição | Modelo | Score Geral | Dimensão Total |',
            '|---------|--------|-------------|----------------|'
        ]
        
        medals = ['🥇', '🥈', '🥉']
        for idx, row in enumerate(sorted_data, 1):
            medal = medals[idx-1] if idx <= 3 else f'{idx}°'
            lines.append(
                f'| {medal} | {row["Model"]} | {row["Overall_Score"]:.2f} | '
                f'{row["Total_Dim"]} |'
            )
        
        lines.extend(['', '---', ''])
        return lines
    
    def _create_classification_section(self, data: List[Dict]) -> List[str]:
        """Cria seção de métricas de classificação."""
        lines = [
            '## 📊 Métricas de Classificação',
            '',
            '| Modelo | Accuracy | F1 | ROC-AUC | Precision | Recall | MCC |',
            '|--------|----------|----|---------|-----------| -------|-----|'
        ]
        
        for row in data:
            lines.append(
                f'| {row["Model"]} | '
                f'{row.get("Accuracy", 0):.3f} | '
                f'{row.get("F1", 0):.3f} | '
                f'{row.get("ROC_AUC", 0):.3f} | '
                f'{row.get("Precision", 0):.3f} | '
                f'{row.get("Recall", 0):.3f} | '
                f'{row.get("MCC", 0):.3f} |'
            )
        
        # Melhor modelo
        best = max(data, key=lambda x: x.get('F1', 0))
        lines.extend([
            '',
            f'**Melhor classificador:** {best["Model"]} '
            f'(F1={best.get("F1", 0):.3f}, ROC-AUC={best.get("ROC_AUC", 0):.3f})',
            '',
            '---',
            ''
        ])
        
        return lines
    
    def _create_regression_section(self, data: List[Dict]) -> List[str]:
        """Cria seção de métricas de regressão."""
        lines = [
            '## 📈 Métricas de Regressão',
            '',
            '### Correlação e Significância',
            '',
            '| Modelo | Pearson R | P-value | RMSE | R² | MAE |',
            '|--------|-----------|---------|------|-----|-----|'
        ]
        
        for row in data:
            # Formatação especial para p-value
            p_value = row.get('Pearson_P', 1)
            p_str = f'{p_value:.2e}' if p_value < 0.001 else f'{p_value:.4f}'
            
            lines.append(
                f'| {row["Model"]} | '
                f'{row.get("Pearson_R", 0):.3f} | '
                f'{p_str} | '
                f'{row.get("RMSE", 0):.3f} | '
                f'{row.get("R2", 0):.3f} | '
                f'{row.get("MAE", 0):.3f} |'
            )
        
        # Análise do melhor modelo (maior correlação positiva)
        best = max(data, key=lambda x: x.get('Pearson_R', -999))  # Buscar maior Pearson (não abs)
        pearson_val = best.get("Pearson_R", 0)
        
        # Interpretar qualidade da correlação
        if pearson_val > 0.7:
            quality = "forte positiva"
        elif pearson_val > 0.3:
            quality = "moderada positiva"
        elif pearson_val > -0.3:
            quality = "fraca/ausente"
        elif pearson_val > -0.7:
            quality = "moderada negativa"
        else:
            quality = "forte negativa"
        
        lines.extend([
            '',
            '### 🎯 Melhor Modelo de Regressão',
            '',
            f'**Modelo:** {best["Model"]}',
            '',
            f'- **Pearson R:** {pearson_val:.3f} ({quality})',
            f'- **P-value:** {best.get("Pearson_P", 1):.2e} '
            f'({"***" if best.get("Pearson_P", 1) < 0.001 else "**" if best.get("Pearson_P", 1) < 0.01 else "*" if best.get("Pearson_P", 1) < 0.05 else "ns"})',
            f'- **RMSE:** {best.get("RMSE", 0):.3f}',
            f'- **R²:** {best.get("R2", 0):.3f}',
            '',
            '---',
            ''
        ])
        
        return lines
    
    def _create_embedding_section(self, data: List[Dict]) -> List[str]:
        """Cria seção de informações de embeddings."""
        lines = [
            '## 🔢 Dimensões de Embeddings',
            '',
            '| Modelo | Proteína | Ligante | Total |',
            '|--------|----------|---------|-------|'
        ]
        
        for row in data:
            lines.append(
                f'| {row["Model"]} | '
                f'{row["Protein_Dim"]} | '
                f'{row["Ligand_Dim"]} | '
                f'{row["Total_Dim"]} |'
            )
        
        lines.extend(['', '---', ''])
        return lines
    
    def _create_conclusions(self, overall_data: List[Dict],
                           classification_data: List[Dict],
                           regression_data: List[Dict]) -> List[str]:
        """Cria seção de conclusões."""
        best_overall = max(overall_data, key=lambda x: x['Overall_Score'])
        best_class = max(classification_data, key=lambda x: x.get('F1', 0))
        best_reg = max(regression_data, key=lambda x: x.get('Pearson_R', -999))  # Maior Pearson positivo
        
        lines = [
            '## 💡 Conclusões',
            '',
            f'### Melhor Modelo Geral: **{best_overall["Model"]}**',
            '',
            f'Score geral de **{best_overall["Overall_Score"]:.2f}**, '
            f'com dimensão total de embedding de {best_overall["Total_Dim"]}.',
            '',
            '### Análise por Tarefa',
            '',
            f'- **Classificação:** {best_class["Model"]} demonstrou melhor '
            f'performance com F1={best_class.get("F1", 0):.3f} e '
            f'ROC-AUC={best_class.get("ROC_AUC", 0):.3f}',
            '',
            f'- **Regressão:** {best_reg["Model"]} obteve a melhor correlação '
            f'(Pearson R={best_reg.get("Pearson_R", 0):.3f}, '
            f'p={best_reg.get("Pearson_P", 1):.2e}) '
            f'com RMSE={best_reg.get("RMSE", 0):.3f}',
            '',
            '### Trade-offs',
            '',
            self._analyze_tradeoffs(overall_data),
            '',
            '---',
            '',
            '*Relatório gerado automaticamente pelo sistema de análise comparativa*'
        ]
        
        return lines
    
    def _analyze_tradeoffs(self, data: List[Dict]) -> str:
        """Analisa trade-offs entre performance e complexidade."""
        sorted_by_score = sorted(data, key=lambda x: x['Overall_Score'], reverse=True)
        sorted_by_dim = sorted(data, key=lambda x: x['Total_Dim'])
        
        most_efficient = sorted_by_dim[0]
        best_performer = sorted_by_score[0]
        
        if most_efficient['Model'] == best_performer['Model']:
            return (f'O modelo **{best_performer["Model"]}** oferece o melhor '
                   f'equilíbrio entre performance e eficiência computacional.')
        else:
            return (f'**{most_efficient["Model"]}** é o mais eficiente '
                   f'(dimensão={most_efficient["Total_Dim"]}), enquanto '
                   f'**{best_performer["Model"]}** oferece a melhor performance '
                   f'(score={best_performer["Overall_Score"]:.2f}).')


def save_dataframes_to_csv(output_dir: Path, 
                           classification_data: List[Dict],
                           regression_data: List[Dict],
                           overall_data: List[Dict]):
    """
    Salva DataFrames em arquivos CSV para análise posterior.
    
    Args:
        output_dir: Diretório de saída
        classification_data: Dados de classificação
        regression_data: Dados de regressão
        overall_data: Dados gerais
    """
    import pandas as pd
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar CSVs
    pd.DataFrame(classification_data).to_csv(
        output_dir / 'classification_metrics.csv', index=False
    )
    pd.DataFrame(regression_data).to_csv(
        output_dir / 'regression_metrics.csv', index=False
    )
    pd.DataFrame(overall_data).to_csv(
        output_dir / 'overall_scores.csv', index=False
    )
