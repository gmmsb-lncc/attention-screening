#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para análise comparativa entre quinases humanas e não humanas.

Este script compara os dados de interação de compostos com quinases humanas
e não humanas, gerando estatísticas e visualizações para entender as diferenças
entre os dois conjuntos de dados.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(human_file, non_human_file):
    """
    Carrega os arquivos TSV de quinases humanas e não humanas.
    
    Args:
        human_file (str): Caminho para o arquivo de quinases humanas
        non_human_file (str): Caminho para o arquivo de quinases não humanas
        
    Returns:
        tuple: DataFrames com dados de humanas e não humanas
    """
    print("Carregando dados...")
    
    # Verificar se os arquivos existem
    if not os.path.exists(human_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {human_file}")
    
    if not os.path.exists(non_human_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {non_human_file}")
    
    # Carregar os dados
    df_human = pd.read_csv(human_file, sep='\t')
    df_non_human = pd.read_csv(non_human_file, sep='\t')
    
    print(f"Quinases humanas: {len(df_human)} registros")
    print(f"Quinases não humanas: {len(df_non_human)} registros")
    
    return df_human, df_non_human

def basic_statistics(df_human, df_non_human):
    """
    Calcula estatísticas básicas dos conjuntos de dados.
    
    Args:
        df_human (pd.DataFrame): DataFrame com dados de quinases humanas
        df_non_human (pd.DataFrame): DataFrame com dados de quinases não humanas
    """
    print("\n=== ESTATÍSTICAS BÁSICAS ===")
    
    # Número de compostos únicos
    human_compounds = df_human['molregno'].nunique()
    non_human_compounds = df_non_human['molregno'].nunique()
    
    print(f"Compostos únicos - Humanas: {human_compounds}")
    print(f"Compostos únicos - Não Humanas: {non_human_compounds}")
    
    # Número de quinases únicas
    human_kinases = df_human['target_kinase'].nunique()
    non_human_kinases = df_non_human['target_kinase'].nunique()
    
    print(f"Quinases únicas - Humanas: {human_kinases}")
    print(f"Quinases únicas - Não Humanas: {non_human_kinases}")
    
    # Organismos mais comuns (para não humanas)
    print("\nOrganismos mais comuns (não humanas):")
    organism_counts = df_non_human['organism'].value_counts().head(10)
    for organism, count in organism_counts.items():
        print(f"  {organism}: {count}")
    
    # Tipos de valores padrão
    print("\nTipos de valores padrão:")
    print("Humanas:", df_human['standard_type'].value_counts().to_dict())
    print("Não Humanas:", df_non_human['standard_type'].value_counts().to_dict())

def activity_distribution(df_human, df_non_human):
    """
    Analisa a distribuição de valores de atividade.
    
    Args:
        df_human (pd.DataFrame): DataFrame com dados de quinases humanas
        df_non_human (pd.DataFrame): DataFrame com dados de quinases não humanas
    """
    print("\n=== DISTRIBUIÇÃO DE ATIVIDADE ===")
    
    # Converter valores para log scale (pIC50)
    df_human['pIC50'] = -np.log10(df_human['standard_value'] * 1e-9)
    df_non_human['pIC50'] = -np.log10(df_non_human['standard_value'] * 1e-9)
    
    # Estatísticas de pIC50
    print("Estatísticas de pIC50:")
    print("Humanas - Média: {:.2f}, Mediana: {:.2f}, Desvio: {:.2f}".format(
        df_human['pIC50'].mean(), df_human['pIC50'].median(), df_human['pIC50'].std()))
    print("Não Humanas - Média: {:.2f}, Mediana: {:.2f}, Desvio: {:.2f}".format(
        df_non_human['pIC50'].mean(), df_non_human['pIC50'].median(), df_non_human['pIC50'].std()))

def compound_overlap_analysis(df_human, df_non_human):
    """
    Analisa a sobreposição de compostos entre humanas e não humanas.
    
    Args:
        df_human (pd.DataFrame): DataFrame com dados de quinases humanas
        df_non_human (pd.DataFrame): DataFrame com dados de quinases não humanas
    """
    print("\n=== ANÁLISE DE SOBREPOSIÇÃO DE COMPOSTOS ===")
    
    human_compounds = set(df_human['molregno'].unique())
    non_human_compounds = set(df_non_human['molregno'].unique())
    
    overlap = human_compounds.intersection(non_human_compounds)
    
    print(f"Compostos que interagem com quinases humanas: {len(human_compounds)}")
    print(f"Compostos que interagem com quinases não humanas: {len(non_human_compounds)}")
    print(f"Compostos que interagem com ambos: {len(overlap)}")
    
    if len(overlap) > 0:
        overlap_percentage_human = len(overlap) / len(human_compounds) * 100
        overlap_percentage_non_human = len(overlap) / len(non_human_compounds) * 100
        
        print(f"Porcentagem de compostos humanos que também estão em não humanos: {overlap_percentage_human:.2f}%")
        print(f"Porcentagem de compostos não humanos que também estão em humanos: {overlap_percentage_non_human:.2f}%")

def kinase_family_analysis(df_human, df_non_human):
    """
    Analisa famílias de quinases presentes em cada conjunto.
    
    Args:
        df_human (pd.DataFrame): DataFrame com dados de quinases humanas
        df_non_human (pd.DataFrame): DataFrame com dados de quinases não humanas
    """
    print("\n=== ANÁLISE DE FAMÍLIAS DE QUINASES ===")
    
    # Extrair famílias de quinases do nome (palavras antes de "kinase")
    def extract_kinase_family(name):
        if pd.isna(name):
            return "Unknown"
        parts = name.split()
        if len(parts) > 1 and parts[-1].lower() == 'kinase':
            return ' '.join(parts[:-1])
        return name
    
    df_human['kinase_family'] = df_human['target_kinase'].apply(extract_kinase_family)
    df_non_human['kinase_family'] = df_non_human['target_kinase'].apply(extract_kinase_family)
    
    # Top famílias
    print("Top 10 famílias de quinases - Humanas:")
    human_families = df_human['kinase_family'].value_counts().head(10)
    for family, count in human_families.items():
        print(f"  {family}: {count}")
    
    print("\nTop 10 famílias de quinases - Não Humanas:")
    non_human_families = df_non_human['kinase_family'].value_counts().head(10)
    for family, count in non_human_families.items():
        print(f"  {family}: {count}")

def generate_visualizations(df_human, df_non_human, output_dir="analysis_output"):
    """
    Gera visualizações comparativas.
    
    Args:
        df_human (pd.DataFrame): DataFrame com dados de quinases humanas
        df_non_human (pd.DataFrame): DataFrame com dados de quinases não humanas
        output_dir (str): Diretório para salvar as visualizações
    """
    print("\n=== GERANDO VISUALIZAÇÕES ===")
    
    # Criar diretório de saída
    Path(output_dir).mkdir(exist_ok=True)
    
    # Converter valores para pIC50
    df_human['pIC50'] = -np.log10(df_human['standard_value'] * 1e-9)
    df_non_human['pIC50'] = -np.log10(df_non_human['standard_value'] * 1e-9)
    
    # 1. Distribuição de pIC50
    plt.figure(figsize=(10, 6))
    plt.hist(df_human['pIC50'].dropna(), bins=50, alpha=0.7, label='Humanas', density=True)
    plt.hist(df_non_human['pIC50'].dropna(), bins=50, alpha=0.7, label='Não Humanas', density=True)
    plt.xlabel('pIC50')
    plt.ylabel('Densidade')
    plt.title('Distribuição de pIC50 - Humanas vs Não Humanas')
    plt.legend()
    plt.savefig(f"{output_dir}/pic50_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Boxplot de pIC50 por tipo de valor padrão
    combined_df = pd.concat([
        df_human[['standard_type', 'pIC50']].assign(source='Humanas'),
        df_non_human[['standard_type', 'pIC50']].assign(source='Não Humanas')
    ])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df, x='standard_type', y='pIC50', hue='source')
    plt.xlabel('Tipo de Valor Padrão')
    plt.ylabel('pIC50')
    plt.title('Distribuição de pIC50 por Tipo de Valor Padrão')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}/pic50_by_standard_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizações salvas em: {output_dir}")

def main():
    """Função principal para execução da análise."""
    # Caminhos para os arquivos (ajuste conforme necessário)
    human_file = "src/database/kinase_human_compounds.tsv"
    non_human_file = "src/database/kinase_non_human_compounds.tsv"
    
    try:
        # Carregar dados
        df_human, df_non_human = load_data(human_file, non_human_file)
        
        # Análises
        basic_statistics(df_human, df_non_human)
        activity_distribution(df_human, df_non_human)
        compound_overlap_analysis(df_human, df_non_human)
        kinase_family_analysis(df_human, df_non_human)
        generate_visualizations(df_human, df_non_human)
        
        print("\n✅ Análise concluída com sucesso!")
        
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
        print("Certifique-se de que os arquivos TSV foram gerados pelos scripts SQL.")
    except Exception as e:
        print(f"❌ Erro durante a análise: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
