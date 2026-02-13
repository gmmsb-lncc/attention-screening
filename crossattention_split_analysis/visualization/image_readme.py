"""Markdown README generator for split-comparison image outputs."""

import os
from typing import Dict, List, Optional


def _path_if_exists(path: str) -> Optional[str]:
    return path if os.path.exists(path) else None


def write_images_readme(
    output_dir: str,
    dataset_type: str,
    seed: int,
    n_folds: int,
    scenarios: List[str],
    split_protocol_version: str,
    affinity_threshold_pchembl: float,
    leakage_artifacts: Dict,
    prefix: str = "",
) -> str:
    """Write README.md describing the logic behind images 01-07."""
    image_06 = _path_if_exists(os.path.join(output_dir, f"{prefix}06_split_comparison.png"))
    image_07 = _path_if_exists(os.path.join(output_dir, f"{prefix}07_inflated_vs_real_performance.png"))

    image_map = {
        "01": leakage_artifacts.get("01_leakage_analysis"),
        "02": leakage_artifacts.get("02_baseline_comparison"),
        "03": leakage_artifacts.get("03_kinase_imbalance"),
        "04": leakage_artifacts.get("04_compound_consistency"),
        "05": leakage_artifacts.get("05_similarity_analysis"),
        "06": image_06,
        "07": image_07,
    }

    lines = []
    lines.append("# README - Logica Das Imagens (Split Comparison Analysis)")
    lines.append("")
    lines.append("Este arquivo documenta **o significado e a logica de calculo** de cada imagem gerada nesta pasta.")
    lines.append("")
    lines.append("## Configuracao Da Execucao")
    lines.append(f"- `dataset`: `{dataset_type}`")
    lines.append(f"- `model_seed`: `{seed}`")
    lines.append(f"- `n_folds`: `{n_folds}`")
    lines.append(f"- `scenarios`: `{scenarios}`")
    lines.append(f"- `split_protocol_version`: `{split_protocol_version}`")
    lines.append(f"- `threshold_pchembl`: `{affinity_threshold_pchembl}`")
    lines.append("")
    lines.append("## Arquivos De Imagem")
    for key in ["01", "02", "03", "04", "05", "06", "07"]:
        value = image_map.get(key)
        status = value if value else "nao gerado nesta execucao"
        lines.append(f"- `{key}`: {status}")
    lines.append("")
    lines.append("## Significado De Cada Imagem")
    lines.append("")
    lines.append("### 01_leakage_analysis.png")
    lines.append("- O que mostra: proporcao de linhas de teste com composto ja visto no treino e proporcao de duplicatas exatas (composto+quinase).")
    lines.append("- Logica: split estratificado; interseccao de `chembl_id` entre treino e teste; interseccao de pares `(chembl_id, target_kinase)`.")
    lines.append("- Interpretacao: valores altos indicam vazamento/memorizacao facilitada.")
    lines.append("")
    lines.append("### 02_baseline_comparison.png")
    lines.append("- O que mostra: comparacao entre baselines de lookup e KNN original (Accuracy e MCC).")
    lines.append("- Logica:")
    lines.append("  - Lookup por composto: classe majoritaria por `chembl_id` no treino.")
    lines.append("  - Lookup por quinase: classe majoritaria por `target_kinase` no treino.")
    lines.append("  - Lookup composto+quinase: cascata par -> composto -> quinase -> classe global.")
    lines.append("  - KNN original: Morgan FP + one-hot de quinase.")
    lines.append("- Interpretacao: se lookup chega perto do KNN, ha forte sinal de memorizacao no dataset.")
    lines.append("")
    lines.append("### 03_kinase_imbalance.png")
    lines.append("- O que mostra: distribuicao da proporcao de ativos por quinase e classes de balanceamento.")
    lines.append("- Logica: para cada quinase, calcula `prop_active`; classifica em desbalanceada/moderada/balanceada.")
    lines.append("- Interpretacao: muitas quinases desbalanceadas tornam a predicao mais facil por vies de classe.")
    lines.append("")
    lines.append("### 04_compound_consistency.png")
    lines.append("- O que mostra: consistencia do comportamento dos compostos em diferentes quinases.")
    lines.append("- Logica: por `chembl_id`, calcula `prop_active` e numero de quinases testadas; marca consistencia perfeita quando `prop_active`=0 ou 1.")
    lines.append("- Interpretacao: compostos muito consistentes carregam grande parte do sinal preditivo.")
    lines.append("")
    lines.append("### 05_similarity_analysis.png")
    lines.append("- O que mostra: similaridade quimica (Tanimoto) de compostos novos de teste vs treino.")
    lines.append("- Logica: Morgan fingerprint; para cada composto novo no teste, calcula a similaridade maxima com o treino.")
    lines.append("- Interpretacao: similaridade alta indica generalizacao quimica curta (teste muito proximo do treino).")
    lines.append("- Observacao: pode nao ser gerada quando nao ha compostos novos suficientes no teste.")
    lines.append("")
    lines.append("### 06_split_comparison.png")
    lines.append("- O que mostra: desempenho de KNN/MLP por cenario de split (Accuracy e MCC).")
    lines.append("- Logica: k-fold com cenarios `random`, `scaffold`, `compound`, `kinase`, `new_compound_new_kinase`; agrega media e desvio-padrao por fold.")
    lines.append("- Interpretacao: quantifica queda de performance conforme o split remove vazamento e aumenta dificuldade de generalizacao.")
    lines.append("")
    lines.append("### 07_inflated_vs_real_performance.png")
    lines.append("- O que mostra: comparacao entre performance inflada (`Random Split`) e performance real de generalizacao (`New Compound + New Kinase`).")
    lines.append("- Logica: barras de MCC e Accuracy para KNN e MLP + percentual de queda.")
    lines.append("- Interpretacao: mede o tamanho da superestimacao quando o protocolo de avaliacao permite vazamento.")
    lines.append("- Observacao: so e gerada quando ambos os cenarios necessarios existem.")
    lines.append("")
    lines.append("## Resumo Da Leitura Recomendada")
    lines.append("1. Comece em `01` para entender vazamento direto.")
    lines.append("2. Use `02-05` para diagnosticar fontes de inflacao (lookup, desbalanceamento, consistencia e similaridade).")
    lines.append("3. Feche com `06-07` para ver o impacto final nos modelos.")
    lines.append("")

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return readme_path
