#!/usr/bin/env python3
"""
Comparação: Split Aleatório vs Split Correto
=============================================

Compara o desempenho de KNN e MLP em três cenários de avaliação:
1. Split Aleatório (Original) - com vazamento de dados
2. Split por Composto - nenhum composto repetido entre treino e teste
3. Novo Composto + Nova Quinase - generalização verdadeira

Objetivo: Demonstrar que a alta performance é artificial devido ao vazamento.

Uso:
    python split_comparison_analysis.py                          # Default: non_human
    python split_comparison_analysis.py --dataset human          # Human dataset
    python split_comparison_analysis.py --dataset non_human      # Non-human dataset
    python split_comparison_analysis.py --dataset all            # All compounds
    python split_comparison_analysis.py --run_all                # Run all datasets sequentially
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

# Dataset paths
DATASET_PATHS = {
    'human': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_human_compounds.tsv',
    'non_human': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_non_human_compounds.tsv',
    'all': '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_all_compounds.tsv'
}

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12


def load_dataset(filepath: str, threshold_nM: float = 1000.0):
    """Carrega o dataset e cria labels binárias."""
    df = pd.read_csv(filepath, sep='\t')
    df['label'] = (df['standard_value'] <= threshold_nM).astype(int)
    return df


def compute_morgan_fingerprints(smiles_list: list, radius: int = 2, n_bits: int = 2048):
    """Computa Morgan fingerprints para lista de SMILES."""
    fingerprints = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
            valid_indices.append(i)

    return np.array(fingerprints), valid_indices


def prepare_features(df: pd.DataFrame, all_kinases: list):
    """Prepara features: Morgan FP + One-Hot Kinase."""
    # Morgan fingerprints
    fps, valid_idx = compute_morgan_fingerprints(df['canonical_smiles'].tolist())

    # One-hot encoding para quinases
    kinase_to_idx = {k: i for i, k in enumerate(all_kinases)}

    def one_hot_kinase(kinase):
        vec = np.zeros(len(all_kinases))
        if kinase in kinase_to_idx:
            vec[kinase_to_idx[kinase]] = 1
        return vec

    kinase_oh = np.array([one_hot_kinase(k) for k in df.iloc[valid_idx]['target_kinase']])
    X = np.hstack([fps, kinase_oh])
    y = df.iloc[valid_idx]['label'].values

    return X, y, valid_idx


# =============================================================================
# CENÁRIO 1: Split Aleatório (Original com vazamento)
# =============================================================================

def split_random(df: pd.DataFrame, random_state: int = 42):
    """Split aleatório estratificado 80/10/10."""
    indices = np.arange(len(df))
    labels = df['label'].values

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=random_state
    )

    return train_idx, val_idx, test_idx


# =============================================================================
# CENÁRIO 2: Split por Composto (sem vazamento de compostos)
# =============================================================================

def split_by_compound(df: pd.DataFrame, random_state: int = 42):
    """
    Split onde nenhum composto do teste aparece no treino.
    Divide primeiro os compostos, depois atribui as linhas.
    """
    # Obter compostos únicos
    compounds = df['chembl_id'].unique()

    # Dividir compostos
    train_compounds, temp_compounds = train_test_split(
        compounds, test_size=0.2, random_state=random_state
    )
    val_compounds, test_compounds = train_test_split(
        temp_compounds, test_size=0.5, random_state=random_state
    )

    # Converter para sets
    train_compounds = set(train_compounds)
    val_compounds = set(val_compounds)
    test_compounds = set(test_compounds)

    # Atribuir linhas aos splits
    train_idx = df[df['chembl_id'].isin(train_compounds)].index.tolist()
    val_idx = df[df['chembl_id'].isin(val_compounds)].index.tolist()
    test_idx = df[df['chembl_id'].isin(test_compounds)].index.tolist()

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# =============================================================================
# CENÁRIO 3: Novo Composto + Nova Quinase (generalização verdadeira)
# =============================================================================

def split_new_compound_new_kinase(df: pd.DataFrame, random_state: int = 42,
                                   test_kinase_frac: float = 0.15,
                                   test_compound_frac: float = 0.15):
    """
    Split onde o teste contém APENAS:
    - Compostos nunca vistos no treino
    - Quinases nunca vistas no treino

    Este é o cenário de generalização verdadeira.
    """
    np.random.seed(random_state)

    # Obter compostos e quinases únicos
    compounds = df['chembl_id'].unique()
    kinases = df['target_kinase'].unique()

    # Separar quinases para teste
    n_test_kinases = max(1, int(len(kinases) * test_kinase_frac))
    test_kinases = set(np.random.choice(kinases, size=n_test_kinases, replace=False))
    train_kinases = set(kinases) - test_kinases

    # Separar compostos para teste
    n_test_compounds = max(1, int(len(compounds) * test_compound_frac))
    test_compounds = set(np.random.choice(compounds, size=n_test_compounds, replace=False))
    train_compounds = set(compounds) - test_compounds

    # Linhas de treino: compostos de treino E quinases de treino
    train_mask = (df['chembl_id'].isin(train_compounds)) & (df['target_kinase'].isin(train_kinases))
    train_idx = df[train_mask].index.tolist()

    # Linhas de teste: compostos de teste E quinases de teste (generalização total)
    test_mask = (df['chembl_id'].isin(test_compounds)) & (df['target_kinase'].isin(test_kinases))
    test_idx = df[test_mask].index.tolist()

    # Validação: amostras restantes (compostos novos com quinases antigas ou vice-versa)
    val_mask = ~train_mask & ~test_mask
    val_idx = df[val_mask].index.tolist()

    # Se teste for muito pequeno, ajustar frações
    if len(test_idx) < 50:
        print(f"  AVISO: Teste muito pequeno ({len(test_idx)}), aumentando frações...")
        return split_new_compound_new_kinase(df, random_state,
                                              test_kinase_frac * 1.5,
                                              test_compound_frac * 1.5)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# =============================================================================
# TREINAMENTO E AVALIAÇÃO
# =============================================================================

def train_and_evaluate(df: pd.DataFrame, train_idx, test_idx, all_kinases: list):
    """Treina KNN e MLP e retorna métricas."""

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Preparar features
    X_train, y_train, train_valid = prepare_features(train_df, all_kinases)
    X_test, y_test, test_valid = prepare_features(test_df, all_kinases)

    if len(X_test) == 0 or len(X_train) == 0:
        return None

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # KNN
    print("    Treinando KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine', n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    results['KNN'] = {
        'accuracy': accuracy_score(y_test, knn_pred),
        'mcc': matthews_corrcoef(y_test, knn_pred)
    }

    # MLP
    print("    Treinando MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    results['MLP'] = {
        'accuracy': accuracy_score(y_test, mlp_pred),
        'mcc': matthews_corrcoef(y_test, mlp_pred)
    }

    return results


# =============================================================================
# ANÁLISE PRINCIPAL
# =============================================================================

def run_comparison(df: pd.DataFrame, output_dir: str = '.'):
    """Executa comparação entre os três cenários de split."""

    print("=" * 70)
    print("COMPARAÇÃO: SPLIT ALEATÓRIO VS SPLIT CORRETO")
    print("=" * 70)

    all_kinases = list(df['target_kinase'].unique())
    all_results = {}
    split_stats = {}

    # CENÁRIO 1: Split Aleatório
    print("\n" + "-" * 50)
    print("CENÁRIO 1: Split Aleatório (com vazamento)")
    print("-" * 50)

    train_idx, val_idx, test_idx = split_random(df)

    # Estatísticas de vazamento
    train_compounds = set(df.iloc[train_idx]['chembl_id'])
    test_compounds = set(df.iloc[test_idx]['chembl_id'])
    leaked = train_compounds & test_compounds

    split_stats['Split Aleatório\n(Original)'] = {
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'test_compounds': len(test_compounds),
        'leaked_compounds': len(leaked),
        'leak_pct': 100 * len(leaked) / len(test_compounds) if test_compounds else 0
    }

    print(f"  Treino: {len(train_idx)} linhas")
    print(f"  Teste:  {len(test_idx)} linhas")
    print(f"  Compostos no teste: {len(test_compounds)}")
    print(f"  Compostos VAZADOS: {len(leaked)} ({split_stats['Split Aleatório\n(Original)']['leak_pct']:.1f}%)")

    results = train_and_evaluate(df, train_idx, test_idx, all_kinases)
    all_results['Split Aleatório\n(Original)'] = results
    print(f"  KNN: Acc={results['KNN']['accuracy']:.4f}, MCC={results['KNN']['mcc']:.4f}")
    print(f"  MLP: Acc={results['MLP']['accuracy']:.4f}, MCC={results['MLP']['mcc']:.4f}")

    # CENÁRIO 2: Split por Composto
    print("\n" + "-" * 50)
    print("CENÁRIO 2: Split por Composto (sem vazamento)")
    print("-" * 50)

    train_idx, val_idx, test_idx = split_by_compound(df)

    train_compounds = set(df.iloc[train_idx]['chembl_id'])
    test_compounds = set(df.iloc[test_idx]['chembl_id'])
    leaked = train_compounds & test_compounds

    split_stats['Split por\nComposto'] = {
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'test_compounds': len(test_compounds),
        'leaked_compounds': len(leaked),
        'leak_pct': 100 * len(leaked) / len(test_compounds) if test_compounds else 0
    }

    print(f"  Treino: {len(train_idx)} linhas")
    print(f"  Teste:  {len(test_idx)} linhas")
    print(f"  Compostos no teste: {len(test_compounds)}")
    print(f"  Compostos VAZADOS: {len(leaked)} ({split_stats['Split por\nComposto']['leak_pct']:.1f}%)")

    results = train_and_evaluate(df, train_idx, test_idx, all_kinases)
    all_results['Split por\nComposto'] = results
    print(f"  KNN: Acc={results['KNN']['accuracy']:.4f}, MCC={results['KNN']['mcc']:.4f}")
    print(f"  MLP: Acc={results['MLP']['accuracy']:.4f}, MCC={results['MLP']['mcc']:.4f}")

    # CENÁRIO 3: Novo Composto + Nova Quinase
    print("\n" + "-" * 50)
    print("CENÁRIO 3: Novo Composto + Nova Quinase (generalização verdadeira)")
    print("-" * 50)

    train_idx, val_idx, test_idx = split_new_compound_new_kinase(df)

    train_compounds = set(df.iloc[train_idx]['chembl_id'])
    test_compounds = set(df.iloc[test_idx]['chembl_id'])
    train_kinases = set(df.iloc[train_idx]['target_kinase'])
    test_kinases = set(df.iloc[test_idx]['target_kinase'])

    leaked_compounds = train_compounds & test_compounds
    leaked_kinases = train_kinases & test_kinases

    split_stats['Novo Comp.\n+ Nova Quinase'] = {
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'test_compounds': len(test_compounds),
        'test_kinases': len(test_kinases),
        'leaked_compounds': len(leaked_compounds),
        'leaked_kinases': len(leaked_kinases),
        'leak_pct': 0  # Por definição
    }

    print(f"  Treino: {len(train_idx)} linhas")
    print(f"  Teste:  {len(test_idx)} linhas")
    print(f"  Compostos no teste: {len(test_compounds)} (todos novos)")
    print(f"  Quinases no teste: {len(test_kinases)} (todas novas)")
    print(f"  Compostos vazados: {len(leaked_compounds)}")
    print(f"  Quinases vazadas: {len(leaked_kinases)}")

    results = train_and_evaluate(df, train_idx, test_idx, all_kinases)
    all_results['Novo Comp.\n+ Nova Quinase'] = results
    print(f"  KNN: Acc={results['KNN']['accuracy']:.4f}, MCC={results['KNN']['mcc']:.4f}")
    print(f"  MLP: Acc={results['MLP']['accuracy']:.4f}, MCC={results['MLP']['mcc']:.4f}")

    return all_results, split_stats




def plot_inflated_vs_real(all_results: dict, output_dir: str = '.', prefix: str = ''):
    """
    Cria gráfico comparando Performance Inflada vs Performance Real.
    Mostra claramente a queda de MCC e Accuracy entre split aleatório e generalização.
    """

    # Extrair métricas
    inflated_key = 'Split Aleatório\n(Original)'
    real_key = 'Novo Comp.\n+ Nova Quinase'

    # Dados para o gráfico
    models = ['KNN', 'MLP']
    metrics = ['MCC', 'Accuracy']

    knn_mcc_inflated = all_results[inflated_key]['KNN']['mcc']
    knn_mcc_real = all_results[real_key]['KNN']['mcc']
    mlp_mcc_inflated = all_results[inflated_key]['MLP']['mcc']
    mlp_mcc_real = all_results[real_key]['MLP']['mcc']

    knn_acc_inflated = all_results[inflated_key]['KNN']['accuracy']
    knn_acc_real = all_results[real_key]['KNN']['accuracy']
    mlp_acc_inflated = all_results[inflated_key]['MLP']['accuracy']
    mlp_acc_real = all_results[real_key]['MLP']['accuracy']

    # Figura com 2 subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # =========================================================================
    # Gráfico 1: MCC - Performance Inflada vs Real
    # =========================================================================
    ax1 = axes[0]

    x = np.arange(2)  # KNN, MLP
    width = 0.35

    inflated_mcc = [knn_mcc_inflated, mlp_mcc_inflated]
    real_mcc = [knn_mcc_real, mlp_mcc_real]

    bars1 = ax1.bar(x - width/2, inflated_mcc, width,
                    label='MCC Reportado (Split Aleatório)',
                    color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, real_mcc, width,
                    label='MCC Real (Generalização)',
                    color='#e74c3c', edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('MCC (Matthews Correlation Coefficient)', fontsize=12)
    ax1.set_title('Performance Inflada vs Performance Real\n(MCC)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['KNN', 'MLP'], fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 1.0)

    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#2c3e50')

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#2c3e50')

    # Adicionar setas e texto mostrando a queda
    for i, (inf, real) in enumerate(zip(inflated_mcc, real_mcc)):
        drop = inf - real
        drop_pct = 100 * drop / inf
        mid_x = i
        mid_y = (inf + real) / 2

        # Seta conectando as barras
        ax1.annotate('',
                    xy=(i + width/2, real + 0.02),
                    xytext=(i - width/2, inf - 0.02),
                    arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.5))

        # Texto da queda
        ax1.text(mid_x, mid_y + 0.08, f'↓ {drop_pct:.0f}%',
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='#8e44ad',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#8e44ad', alpha=0.9))

    # Linha de referência
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(1.5, 0.52, 'MCC = 0.5', fontsize=9, color='gray', ha='right')

    # =========================================================================
    # Gráfico 2: Accuracy - Performance Inflada vs Real
    # =========================================================================
    ax2 = axes[1]

    inflated_acc = [knn_acc_inflated, mlp_acc_inflated]
    real_acc = [knn_acc_real, mlp_acc_real]

    bars3 = ax2.bar(x - width/2, inflated_acc, width,
                    label='Accuracy Reportada (Split Aleatório)',
                    color='#3498db', edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, real_acc, width,
                    label='Accuracy Real (Generalização)',
                    color='#e74c3c', edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Performance Inflada vs Performance Real\n(Accuracy)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['KNN', 'MLP'], fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 1.0)

    # Adicionar valores nas barras
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#2c3e50')

    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#2c3e50')

    # Adicionar setas e texto mostrando a queda
    for i, (inf, real) in enumerate(zip(inflated_acc, real_acc)):
        drop = inf - real
        drop_pct = 100 * drop / inf
        mid_x = i
        mid_y = (inf + real) / 2

        # Seta conectando as barras
        ax2.annotate('',
                    xy=(i + width/2, real + 0.02),
                    xytext=(i - width/2, inf - 0.02),
                    arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.5))

        # Texto da queda
        ax2.text(mid_x, mid_y + 0.06, f'↓ {drop_pct:.0f}%',
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='#8e44ad',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#8e44ad', alpha=0.9))

    # Linha de referência
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(1.5, 0.52, 'Random = 0.5', fontsize=9, color='gray', ha='right')

    plt.tight_layout()
    filename = f'{prefix}07_inflated_vs_real_performance.png' if prefix else '07_inflated_vs_real_performance.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo: {output_dir}/{filename}")


# =============================================================================
# MAIN
# =============================================================================

def run_single_dataset(dataset_type: str, output_dir: str):
    """Run analysis for a single dataset type."""
    data_path = DATASET_PATHS.get(dataset_type)
    if not data_path:
        print(f"Error: Unknown dataset type '{dataset_type}'")
        return None

    print("\n" + "=" * 70)
    print(f"ANÁLISE: {dataset_type.upper()}")
    print("=" * 70)

    # Carregar dados
    print(f"\nCarregando dataset: {dataset_type}...")
    try:
        df = load_dataset(data_path)
    except FileNotFoundError:
        print(f"  ERRO: Arquivo não encontrado: {data_path}")
        return None

    print(f"  Total: {len(df)} linhas, {df['chembl_id'].nunique()} compostos, {df['target_kinase'].nunique()} quinases")

    # Executar comparação
    all_results, split_stats = run_comparison(df, output_dir)

    # Gerar gráficos com prefixo do dataset
    print("\n" + "-" * 50)
    print("GERANDO GRÁFICOS")
    print("-" * 50)

    # Usar prefixo para diferenciar os arquivos
    prefix = f"{dataset_type}_" if dataset_type != 'non_human' else ""
    plot_comparison(all_results, split_stats, output_dir, prefix)

    # Conclusão
    print("\n" + "-" * 50)
    print(f"CONCLUSÃO ({dataset_type.upper()})")
    print("-" * 50)

    orig_knn_mcc = all_results['Split Aleatório\n(Original)']['KNN']['mcc']
    orig_mlp_mcc = all_results['Split Aleatório\n(Original)']['MLP']['mcc']
    new_knn_mcc = all_results['Novo Comp.\n+ Nova Quinase']['KNN']['mcc']
    new_mlp_mcc = all_results['Novo Comp.\n+ Nova Quinase']['MLP']['mcc']

    print(f"\nQueda de MCC do KNN:  {orig_knn_mcc:.3f} → {new_knn_mcc:.3f} (Δ = {new_knn_mcc - orig_knn_mcc:+.3f})")
    print(f"Queda de MCC do MLP:  {orig_mlp_mcc:.3f} → {new_mlp_mcc:.3f} (Δ = {new_mlp_mcc - orig_mlp_mcc:+.3f})")

    return all_results


def plot_comparison(all_results: dict, split_stats: dict, output_dir: str = '.', prefix: str = ''):
    """Gera gráficos comparativos."""

    scenarios = list(all_results.keys())

    # Extrair métricas
    knn_acc = [all_results[s]['KNN']['accuracy'] for s in scenarios]
    knn_mcc = [all_results[s]['KNN']['mcc'] for s in scenarios]
    mlp_acc = [all_results[s]['MLP']['accuracy'] for s in scenarios]
    mlp_mcc = [all_results[s]['MLP']['mcc'] for s in scenarios]

    # Tamanhos dos conjuntos de teste
    test_sizes = [split_stats[s]['test_size'] for s in scenarios]
    test_compounds = [split_stats[s]['test_compounds'] for s in scenarios]

    # Figura com 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    x = np.arange(len(scenarios))
    width = 0.35

    # Gráfico 1: Acurácia
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, knn_acc, width, label='KNN', color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x + width/2, mlp_acc, width, label='MLP', color='#e74c3c', edgecolor='black')

    ax1.set_ylabel('Acurácia', fontsize=12)
    ax1.set_title('Acurácia por Cenário de Avaliação', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_ylim(0, 1.15)

    # Valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Adicionar tamanho do teste abaixo do eixo X
    for i, (ts, tc) in enumerate(zip(test_sizes, test_compounds)):
        ax1.annotate(f'n={ts}\n({tc} comp.)',
                    xy=(i, 0), xytext=(0, -25),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, color='gray')

    # Gráfico 2: MCC
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, knn_mcc, width, label='KNN', color='#3498db', edgecolor='black')
    bars4 = ax2.bar(x + width/2, mlp_mcc, width, label='MLP', color='#e74c3c', edgecolor='black')

    ax2.set_ylabel('MCC', fontsize=12)
    ax2.set_title('MCC por Cenário de Avaliação', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_ylim(0, 1.0)

    # Valores nas barras
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Adicionar tamanho do teste abaixo do eixo X
    for i, (ts, tc) in enumerate(zip(test_sizes, test_compounds)):
        ax2.annotate(f'n={ts}\n({tc} comp.)',
                    xy=(i, 0), xytext=(0, -25),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    filename = f'{prefix}06_split_comparison.png' if prefix else '06_split_comparison.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico salvo: {output_dir}/{filename}")

    # Gráfico adicional: Performance Inflada vs Real
    plot_inflated_vs_real(all_results, output_dir, prefix)


def main():
    parser = argparse.ArgumentParser(description='Comparação de splits para análise de vazamento')
    parser.add_argument('--dataset', type=str, default='non_human',
                        choices=['human', 'non_human', 'all'],
                        help='Tipo de dataset (human, non_human, all)')
    parser.add_argument('--run_all', action='store_true',
                        help='Executar análise para todos os datasets sequencialmente')
    parser.add_argument('--output_dir', type=str,
                        default='/media/leon/ssd2tb/docktkinase/leakage_analysis_results',
                        help='Diretório de saída')

    args = parser.parse_args()

    if args.run_all:
        print("\n" + "=" * 70)
        print("EXECUTANDO ANÁLISE PARA TODOS OS DATASETS")
        print("=" * 70)

        for dataset_type in ['non_human', 'human', 'all']:
            run_single_dataset(dataset_type, args.output_dir)

        print("\n" + "=" * 70)
        print("TODAS AS ANÁLISES CONCLUÍDAS!")
        print("=" * 70)
    else:
        run_single_dataset(args.dataset, args.output_dir)

    print(f"\nResultados salvos em: {args.output_dir}")


if __name__ == '__main__':
    main()
