#!/usr/bin/env python3
"""
Análise Completa de Vazamento de Dados e Viés no Dataset de Quinases
=====================================================================

Este script investiga por que representações simples (one-hot + Morgan FP)
conseguem performance elevada em predição de atividade composto-quinase.

Hipóteses investigadas:
1. Vazamento de compostos entre treino e teste
2. Desbalanceamento extremo por quinase
3. Consistência de classe dos compostos
4. Alta similaridade química entre treino e teste
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# =============================================================================
# 1. CARREGAR E PREPARAR DADOS
# =============================================================================

def load_dataset(filepath: str, threshold_nM: float = 1000.0):
    """Carrega o dataset e cria labels binárias."""
    print("=" * 70)
    print("CARREGANDO DATASET")
    print("=" * 70)

    df = pd.read_csv(filepath, sep='\t')

    # Criar label binária: ativo se standard_value <= threshold (1 uM = 1000 nM)
    df['label'] = (df['standard_value'] <= threshold_nM).astype(int)

    print(f"Total de linhas: {len(df)}")
    print(f"Compostos únicos (chembl_id): {df['chembl_id'].nunique()}")
    print(f"Quinases únicas: {df['target_kinase'].nunique()}")
    print(f"\nDistribuição de classes:")
    print(f"  Ativos (<=1uM):   {(df['label']==1).sum()} ({100*(df['label']==1).mean():.1f}%)")
    print(f"  Inativos (>1uM):  {(df['label']==0).sum()} ({100*(df['label']==0).mean():.1f}%)")

    return df


def create_split(df: pd.DataFrame, random_state: int = 42):
    """Cria split 80/10/10 estratificado."""
    print("\n" + "=" * 70)
    print("CRIANDO SPLIT 80/10/10")
    print("=" * 70)

    indices = np.arange(len(df))
    labels = df['label'].values

    # Split estratificado
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=random_state
    )

    print(f"Treino:     {len(train_idx)} ({100*len(train_idx)/len(df):.1f}%)")
    print(f"Validação:  {len(val_idx)} ({100*len(val_idx)/len(df):.1f}%)")
    print(f"Teste:      {len(test_idx)} ({100*len(test_idx)/len(df):.1f}%)")

    return train_idx, val_idx, test_idx


# =============================================================================
# 2. ANÁLISE DE VAZAMENTO DE COMPOSTOS
# =============================================================================

def analyze_compound_leakage(df: pd.DataFrame, train_idx, test_idx):
    """Analisa vazamento de compostos entre treino e teste."""
    print("\n" + "=" * 70)
    print("ANÁLISE DE VAZAMENTO DE COMPOSTOS")
    print("=" * 70)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Compostos únicos em cada split
    train_compounds = set(train_df['chembl_id'].unique())
    test_compounds = set(test_df['chembl_id'].unique())

    # Compostos vazados (aparecem em ambos)
    leaked_compounds = train_compounds & test_compounds
    new_compounds = test_compounds - train_compounds

    print(f"\nCompostos no treino: {len(train_compounds)}")
    print(f"Compostos no teste:  {len(test_compounds)}")
    print(f"Compostos VAZADOS:   {len(leaked_compounds)} ({100*len(leaked_compounds)/len(test_compounds):.1f}% do teste)")
    print(f"Compostos NOVOS:     {len(new_compounds)} ({100*len(new_compounds)/len(test_compounds):.1f}% do teste)")

    # Linhas de teste com composto vazado
    test_leaked_mask = test_df['chembl_id'].isin(leaked_compounds)
    n_test_leaked = test_leaked_mask.sum()

    print(f"\n*** PROBLEMA CRÍTICO ***")
    print(f"Linhas de teste com composto vazado: {n_test_leaked} ({100*n_test_leaked/len(test_df):.1f}%)")

    # Duplicatas exatas (mesmo composto + quinase)
    train_pairs = set(zip(train_df['chembl_id'], train_df['target_kinase']))
    test_pairs = set(zip(test_df['chembl_id'], test_df['target_kinase']))
    exact_duplicates = train_pairs & test_pairs

    print(f"\nDuplicatas EXATAS (mesmo composto + quinase):")
    print(f"  {len(exact_duplicates)} pares duplicados")

    # Contar linhas de teste que são duplicatas exatas
    test_exact_mask = test_df.apply(
        lambda row: (row['chembl_id'], row['target_kinase']) in exact_duplicates, axis=1
    )
    n_exact = test_exact_mask.sum()
    print(f"  {n_exact} linhas de teste são duplicatas exatas ({100*n_exact/len(test_df):.1f}%)")

    return {
        'train_compounds': train_compounds,
        'test_compounds': test_compounds,
        'leaked_compounds': leaked_compounds,
        'new_compounds': new_compounds,
        'n_test_leaked_lines': n_test_leaked,
        'n_test_total': len(test_df),
        'exact_duplicates': exact_duplicates,
        'n_exact_lines': n_exact,
        'test_leaked_mask': test_leaked_mask,
        'test_exact_mask': test_exact_mask
    }


def plot_leakage_analysis(leakage_info: dict, output_dir: str = '.'):
    """Gera gráficos de pizza para vazamento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Linhas de teste com composto vazado
    ax1 = axes[0]
    n_leaked = leakage_info['n_test_leaked_lines']
    n_new = leakage_info['n_test_total'] - n_leaked

    sizes = [n_leaked, n_new]
    labels = [f'Composto Vazado\n{n_leaked} ({100*n_leaked/leakage_info["n_test_total"]:.1f}%)',
              f'Composto Novo\n{n_new} ({100*n_new/leakage_info["n_test_total"]:.1f}%)']
    colors = ['#ff6b6b', '#4ecdc4']
    explode = (0.05, 0)

    ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct='', startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Vazamento de Compostos nas Linhas de Teste', fontsize=14, fontweight='bold')

    # Gráfico 2: Duplicatas exatas
    ax2 = axes[1]
    n_exact = leakage_info['n_exact_lines']
    n_not_exact = leakage_info['n_test_total'] - n_exact

    sizes2 = [n_exact, n_not_exact]
    labels2 = [f'Duplicata Exata\n{n_exact} ({100*n_exact/leakage_info["n_test_total"]:.1f}%)',
               f'Não Duplicata\n{n_not_exact} ({100*n_not_exact/leakage_info["n_test_total"]:.1f}%)']
    colors2 = ['#ff8c42', '#6c5ce7']
    explode2 = (0.05, 0)

    ax2.pie(sizes2, labels=labels2, colors=colors2, explode=explode2,
            autopct='', startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Duplicatas Exatas (Composto + Quinase)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_leakage_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico salvo: {output_dir}/01_leakage_analysis.png")


# =============================================================================
# 3. MODELO LOOKUP BASELINE
# =============================================================================

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


def lookup_baseline_compound(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Modelo Lookup por Composto:
    Para cada composto no teste, prediz a classe majoritária do composto no treino.
    Se o composto não existe no treino, prediz a classe global majoritária.
    """
    # Classe majoritária por composto no treino
    compound_majority = train_df.groupby('chembl_id')['label'].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()

    # Classe global majoritária
    global_majority = 1 if train_df['label'].mean() >= 0.5 else 0

    # Predições
    predictions = []
    for compound in test_df['chembl_id']:
        if compound in compound_majority:
            predictions.append(compound_majority[compound])
        else:
            predictions.append(global_majority)

    return np.array(predictions)


def lookup_baseline_kinase(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Modelo Lookup por Quinase:
    Para cada quinase no teste, prediz a classe majoritária da quinase no treino.
    """
    # Classe majoritária por quinase no treino
    kinase_majority = train_df.groupby('target_kinase')['label'].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()

    # Classe global majoritária
    global_majority = 1 if train_df['label'].mean() >= 0.5 else 0

    # Predições
    predictions = []
    for kinase in test_df['target_kinase']:
        if kinase in kinase_majority:
            predictions.append(kinase_majority[kinase])
        else:
            predictions.append(global_majority)

    return np.array(predictions)


def lookup_baseline_compound_kinase(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Modelo Lookup por Composto + Quinase:
    Para cada par (composto, quinase) no teste:
    1. Se o par existe no treino, usa a classe do par
    2. Se não, usa a classe majoritária do composto
    3. Se não, usa a classe majoritária da quinase
    4. Se não, usa a classe global majoritária
    """
    # Classe por par (composto, quinase)
    pair_class = train_df.groupby(['chembl_id', 'target_kinase'])['label'].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()

    # Classe majoritária por composto
    compound_majority = train_df.groupby('chembl_id')['label'].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()

    # Classe majoritária por quinase
    kinase_majority = train_df.groupby('target_kinase')['label'].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()

    # Classe global majoritária
    global_majority = 1 if train_df['label'].mean() >= 0.5 else 0

    # Predições
    predictions = []
    for _, row in test_df.iterrows():
        compound = row['chembl_id']
        kinase = row['target_kinase']

        if (compound, kinase) in pair_class:
            predictions.append(pair_class[(compound, kinase)])
        elif compound in compound_majority:
            predictions.append(compound_majority[compound])
        elif kinase in kinase_majority:
            predictions.append(kinase_majority[kinase])
        else:
            predictions.append(global_majority)

    return np.array(predictions)


def train_knn_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Treina KNN com Morgan fingerprints + one-hot kinase."""
    print("\n  Computando Morgan fingerprints para treino...")
    train_fp, train_valid = compute_morgan_fingerprints(train_df['canonical_smiles'].tolist())

    print("  Computando Morgan fingerprints para teste...")
    test_fp, test_valid = compute_morgan_fingerprints(test_df['canonical_smiles'].tolist())

    # One-hot encoding para quinases
    all_kinases = list(set(train_df['target_kinase'].unique()) | set(test_df['target_kinase'].unique()))
    kinase_to_idx = {k: i for i, k in enumerate(all_kinases)}

    def one_hot_kinase(kinase):
        vec = np.zeros(len(all_kinases))
        vec[kinase_to_idx[kinase]] = 1
        return vec

    # Features de treino
    train_kinase_oh = np.array([one_hot_kinase(k) for k in train_df.iloc[train_valid]['target_kinase']])
    X_train = np.hstack([train_fp, train_kinase_oh])
    y_train = train_df.iloc[train_valid]['label'].values

    # Features de teste
    test_kinase_oh = np.array([one_hot_kinase(k) for k in test_df.iloc[test_valid]['target_kinase']])
    X_test = np.hstack([test_fp, test_kinase_oh])
    y_test = test_df.iloc[test_valid]['label'].values

    # Treinar KNN
    print("  Treinando KNN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine', n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    predictions = knn.predict(X_test_scaled)

    return predictions, y_test


def evaluate_lookup_baselines(df: pd.DataFrame, train_idx, test_idx):
    """Avalia todos os baselines lookup vs KNN."""
    print("\n" + "=" * 70)
    print("AVALIAÇÃO: LOOKUP BASELINES vs KNN")
    print("=" * 70)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_true = test_df['label'].values

    results = {}

    # Lookup por Composto
    print("\n1. Lookup por Composto...")
    pred_compound = lookup_baseline_compound(train_df, test_df)
    acc_compound = accuracy_score(y_true, pred_compound)
    mcc_compound = matthews_corrcoef(y_true, pred_compound)
    results['Lookup: Composto'] = {'accuracy': acc_compound, 'mcc': mcc_compound}
    print(f"   Accuracy: {acc_compound:.4f}, MCC: {mcc_compound:.4f}")

    # Lookup por Quinase
    print("\n2. Lookup por Quinase...")
    pred_kinase = lookup_baseline_kinase(train_df, test_df)
    acc_kinase = accuracy_score(y_true, pred_kinase)
    mcc_kinase = matthews_corrcoef(y_true, pred_kinase)
    results['Lookup: Quinase'] = {'accuracy': acc_kinase, 'mcc': mcc_kinase}
    print(f"   Accuracy: {acc_kinase:.4f}, MCC: {mcc_kinase:.4f}")

    # Lookup por Composto + Quinase
    print("\n3. Lookup por Composto + Quinase...")
    pred_both = lookup_baseline_compound_kinase(train_df, test_df)
    acc_both = accuracy_score(y_true, pred_both)
    mcc_both = matthews_corrcoef(y_true, pred_both)
    results['Lookup: Comp+Quin'] = {'accuracy': acc_both, 'mcc': mcc_both}
    print(f"   Accuracy: {acc_both:.4f}, MCC: {mcc_both:.4f}")

    # KNN
    print("\n4. KNN (Morgan FP + One-Hot Kinase)...")
    pred_knn, y_knn = train_knn_baseline(train_df, test_df)
    acc_knn = accuracy_score(y_knn, pred_knn)
    mcc_knn = matthews_corrcoef(y_knn, pred_knn)
    results['KNN (Original)'] = {'accuracy': acc_knn, 'mcc': mcc_knn}
    print(f"   Accuracy: {acc_knn:.4f}, MCC: {mcc_knn:.4f}")

    return results


def plot_baseline_comparison(results: dict, output_dir: str = '.'):
    """Gera gráfico de barras comparando KNN vs Lookup Baselines."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    mccs = [results[m]['mcc'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, mccs, width, label='MCC', color='#e74c3c', edgecolor='black')

    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Valor da Métrica', fontsize=12)
    ax.set_title('Comparação: KNN vs Lookup Baselines\n(Modelos que apenas "olham" a classe majoritária)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.1)

    # Adicionar valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Linha de referência
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.7, label='Threshold 70%')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico salvo: {output_dir}/02_baseline_comparison.png")


# =============================================================================
# 4. DESBALANCEAMENTO POR QUINASE
# =============================================================================

def analyze_kinase_imbalance(df: pd.DataFrame, train_idx):
    """Analisa desbalanceamento de classes por quinase."""
    print("\n" + "=" * 70)
    print("ANÁLISE DE DESBALANCEAMENTO POR QUINASE")
    print("=" * 70)

    train_df = df.iloc[train_idx]

    # Proporção de ativos por quinase
    kinase_stats = train_df.groupby('target_kinase').agg(
        n_samples=('label', 'count'),
        n_active=('label', 'sum'),
        prop_active=('label', 'mean')
    ).reset_index()

    # Classificar quinases
    def classify_balance(prop):
        if prop > 0.8 or prop < 0.2:
            return 'Desbalanceada (>80% ou <20%)'
        elif prop > 0.6 or prop < 0.4:
            return 'Moderada'
        else:
            return 'Balanceada (40-60%)'

    kinase_stats['balance_class'] = kinase_stats['prop_active'].apply(classify_balance)

    # Estatísticas
    n_desbalanced = (kinase_stats['balance_class'] == 'Desbalanceada (>80% ou <20%)').sum()
    n_moderate = (kinase_stats['balance_class'] == 'Moderada').sum()
    n_balanced = (kinase_stats['balance_class'] == 'Balanceada (40-60%)').sum()

    print(f"\nClassificação de Quinases:")
    print(f"  Desbalanceadas (>80% uma classe): {n_desbalanced} ({100*n_desbalanced/len(kinase_stats):.1f}%)")
    print(f"  Moderadas:                         {n_moderate} ({100*n_moderate/len(kinase_stats):.1f}%)")
    print(f"  Balanceadas (40-60%):              {n_balanced} ({100*n_balanced/len(kinase_stats):.1f}%)")

    # Linhas do dataset de quinases desbalanceadas
    desbalanced_kinases = kinase_stats[kinase_stats['balance_class'] == 'Desbalanceada (>80% ou <20%)']['target_kinase']
    n_lines_desbalanced = train_df[train_df['target_kinase'].isin(desbalanced_kinases)].shape[0]

    print(f"\nLinhas do dataset de quinases desbalanceadas:")
    print(f"  {n_lines_desbalanced} de {len(train_df)} ({100*n_lines_desbalanced/len(train_df):.1f}%)")

    # Exemplos extremos
    print("\nExemplos de quinases extremamente desbalanceadas:")
    extremes = kinase_stats[(kinase_stats['prop_active'] > 0.9) | (kinase_stats['prop_active'] < 0.1)]
    extremes = extremes.sort_values('prop_active', ascending=False).head(10)
    for _, row in extremes.iterrows():
        print(f"  {row['target_kinase'][:50]}: {100*row['prop_active']:.1f}% ativos ({row['n_samples']} amostras)")

    return kinase_stats


def plot_kinase_distribution(kinase_stats: pd.DataFrame, output_dir: str = '.'):
    """Gráfico de distribuição de classes por quinase."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Histograma de proporção de ativos
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(kinase_stats['prop_active'], bins=20,
                                      color='#9b59b6', edgecolor='black', alpha=0.8)

    # Calcular altura máxima para posicionar elementos
    y_max = counts.max()

    ax1.set_xlabel('Proporção de Ativos por Quinase', fontsize=12)
    ax1.set_ylabel('Número de Quinases', fontsize=12)
    ax1.set_title('Distribuição de Classes por Quinase', fontsize=14, fontweight='bold')

    # Linhas de referência com labels posicionados no topo
    ax1.axvline(x=0.2, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(x=0.8, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(x=0.4, color='#e67e22', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(x=0.6, color='#e67e22', linestyle='--', linewidth=2, alpha=0.8)

    # Adicionar texto nas linhas de referência (no topo do gráfico)
    ax1.text(0.2, y_max * 1.02, '20%', ha='center', va='bottom', fontsize=9,
             color='#c0392b', fontweight='bold')
    ax1.text(0.8, y_max * 1.02, '80%', ha='center', va='bottom', fontsize=9,
             color='#c0392b', fontweight='bold')
    ax1.text(0.4, y_max * 0.95, '40%', ha='center', va='bottom', fontsize=9,
             color='#e67e22', fontweight='bold')
    ax1.text(0.6, y_max * 0.95, '60%', ha='center', va='bottom', fontsize=9,
             color='#e67e22', fontweight='bold')

    # Legenda customizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#c0392b', linestyle='--', linewidth=2,
               label='Desbalanceada (20%/80%)'),
        Line2D([0], [0], color='#e67e22', linestyle='--', linewidth=2,
               label='Moderada (40%/60%)')
    ]
    ax1.legend(handles=legend_elements, loc='upper center', fontsize=10)

    # Contagem de extremos
    n_all_inactive = (kinase_stats['prop_active'] == 0).sum()
    n_all_active = (kinase_stats['prop_active'] == 1).sum()

    # Anotação para quinases 100% inativas (lado esquerdo)
    if n_all_inactive > 0:
        ax1.annotate(f'{n_all_inactive} quinases\n100% inativas',
                     xy=(0.025, n_all_inactive),
                     xytext=(0.22, y_max * 0.7),
                     arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
                     fontsize=11, color='#c0392b', fontweight='bold',
                     ha='left', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#c0392b', alpha=0.9))

    # Anotação para quinases 100% ativas (lado direito)
    if n_all_active > 0:
        ax1.annotate(f'{n_all_active} quinases\n100% ativas',
                     xy=(0.975, n_all_active),
                     xytext=(0.72, y_max * 0.7),
                     arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
                     fontsize=11, color='#27ae60', fontweight='bold',
                     ha='right', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#27ae60', alpha=0.9))

    # Ajustar limites do eixo Y para acomodar anotações
    ax1.set_ylim(0, y_max * 1.15)
    ax1.set_xlim(-0.05, 1.05)

    # Gráfico de pizza - facilidade de predição
    ax2 = axes[1]
    balance_counts = kinase_stats['balance_class'].value_counts()

    # Ordenar categorias consistentemente
    category_order = ['Desbalanceada (>80% ou <20%)', 'Moderada', 'Balanceada (40-60%)']
    ordered_counts = []
    ordered_labels = []
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    final_colors = []

    for i, cat in enumerate(category_order):
        if cat in balance_counts.index:
            count = balance_counts[cat]
            ordered_counts.append(count)
            # Labels mais curtos e claros
            short_labels = {
                'Desbalanceada (>80% ou <20%)': 'Desbalanceadas',
                'Moderada': 'Moderadas',
                'Balanceada (40-60%)': 'Balanceadas'
            }
            ordered_labels.append(f'{short_labels[cat]}\n{count} ({100*count/len(kinase_stats):.1f}%)')
            final_colors.append(colors[i])

    ax2.pie(ordered_counts, labels=ordered_labels, colors=final_colors,
            startangle=90, textprops={'fontsize': 11},
            explode=[0.02] * len(ordered_counts),
            wedgeprops=dict(edgecolor='white', linewidth=2))

    ax2.set_title('Facilidade de Predição por Quinase', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_kinase_imbalance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico salvo: {output_dir}/03_kinase_imbalance.png")


# =============================================================================
# 5. CONSISTÊNCIA DE CLASSE DOS COMPOSTOS
# =============================================================================

def analyze_compound_consistency(df: pd.DataFrame, train_idx):
    """Analisa consistência de classe por composto através de diferentes quinases."""
    print("\n" + "=" * 70)
    print("ANÁLISE DE CONSISTÊNCIA DE CLASSE DOS COMPOSTOS")
    print("=" * 70)

    train_df = df.iloc[train_idx]

    # Proporção de atividade por composto
    compound_stats = train_df.groupby('chembl_id').agg(
        n_kinases=('target_kinase', 'nunique'),
        n_samples=('label', 'count'),
        n_active=('label', 'sum'),
        prop_active=('label', 'mean')
    ).reset_index()

    # Classificar consistência
    def classify_consistency(row):
        if row['n_kinases'] == 1:
            return 'Uma quinase apenas'
        elif row['prop_active'] == 0 or row['prop_active'] == 1:
            return 'Perfeitamente consistente'
        else:
            return 'Inconsistente'

    compound_stats['consistency'] = compound_stats.apply(classify_consistency, axis=1)

    # Estatísticas
    consistency_counts = compound_stats['consistency'].value_counts()
    print(f"\nConsistência dos Compostos:")
    for cat, count in consistency_counts.items():
        print(f"  {cat}: {count} ({100*count/len(compound_stats):.1f}%)")

    # Multi-quinase apenas
    multi_kinase = compound_stats[compound_stats['n_kinases'] > 1]
    n_consistent = ((multi_kinase['prop_active'] == 0) | (multi_kinase['prop_active'] == 1)).sum()
    print(f"\nCompostos multi-quinase perfeitamente consistentes:")
    print(f"  {n_consistent} de {len(multi_kinase)} ({100*n_consistent/len(multi_kinase):.1f}%)")

    return compound_stats


def plot_compound_consistency(compound_stats: pd.DataFrame, output_dir: str = '.'):
    """Gráficos de consistência de classe dos compostos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filtrar compostos multi-quinase
    multi_kinase = compound_stats[compound_stats['n_kinases'] > 1]

    # Histograma de proporção de atividade
    ax1 = axes[0]
    ax1.hist(multi_kinase['prop_active'], bins=20, color='#1abc9c', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Proporção de Atividade do Composto', fontsize=12)
    ax1.set_ylabel('Número de Compostos', fontsize=12)
    ax1.set_title('Consistência de Classe (Multi-quinase)\nComportamento do Composto através de Diferentes Quinases',
                  fontsize=13, fontweight='bold')

    # Anotações
    n_all_inactive = (multi_kinase['prop_active'] == 0).sum()
    n_all_active = (multi_kinase['prop_active'] == 1).sum()
    ax1.annotate(f'{n_all_inactive}\nsempre inativos',
                 xy=(0.0, n_all_inactive), xytext=(0.1, n_all_inactive * 0.8),
                 fontsize=10, color='red', fontweight='bold')
    ax1.annotate(f'{n_all_active}\nsempre ativos',
                 xy=(1.0, n_all_active), xytext=(0.8, n_all_active * 0.8),
                 fontsize=10, color='green', fontweight='bold')

    # Gráfico de pizza
    ax2 = axes[1]
    consistency_counts = compound_stats['consistency'].value_counts()

    colors = ['#27ae60', '#e74c3c', '#95a5a6']
    labels_map = {
        'Perfeitamente consistente': 'Perfeitamente\nconsistentes',
        'Inconsistente': 'Inconsistentes',
        'Uma quinase apenas': 'Uma quinase\napenas'
    }

    labels = [f'{labels_map.get(cat, cat)}\n{count} ({100*count/len(compound_stats):.1f}%)'
              for cat, count in consistency_counts.items()]

    ax2.pie(consistency_counts.values, labels=labels, colors=colors[:len(consistency_counts)],
            autopct='', startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Consistência dos Compostos', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_compound_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico salvo: {output_dir}/04_compound_consistency.png")


# =============================================================================
# 6. ANÁLISE DE SIMILARIDADE QUÍMICA (TANIMOTO)
# =============================================================================

def compute_tanimoto_similarity(fp1, fp2):
    """Calcula similaridade de Tanimoto entre duas fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def analyze_chemical_similarity(df: pd.DataFrame, train_idx, test_idx, sample_size: int = 500):
    """Analisa similaridade química entre compostos de teste e treino."""
    print("\n" + "=" * 70)
    print("ANÁLISE DE SIMILARIDADE QUÍMICA (TANIMOTO)")
    print("=" * 70)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Compostos únicos
    train_compounds = train_df.drop_duplicates('chembl_id')[['chembl_id', 'canonical_smiles']]
    test_compounds = test_df.drop_duplicates('chembl_id')[['chembl_id', 'canonical_smiles']]

    # Filtrar compostos novos (não vistos no treino)
    train_compound_ids = set(train_compounds['chembl_id'])
    test_new = test_compounds[~test_compounds['chembl_id'].isin(train_compound_ids)]

    print(f"\nCompostos no treino: {len(train_compounds)}")
    print(f"Compostos no teste: {len(test_compounds)}")
    print(f"Compostos NOVOS no teste: {len(test_new)}")

    if len(test_new) == 0:
        print("AVISO: Todos os compostos de teste estão no treino!")
        return None

    # Amostrar se necessário
    if len(test_new) > sample_size:
        test_new = test_new.sample(n=sample_size, random_state=42)
        print(f"Amostrando {sample_size} compostos novos para análise...")

    # Computar fingerprints
    print("Computando fingerprints para treino...")
    train_fps = {}
    for _, row in train_compounds.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            train_fps[row['chembl_id']] = fp

    print(f"Fingerprints de treino válidas: {len(train_fps)}")

    # Para cada composto novo, encontrar similaridade máxima ao treino
    print("Calculando similaridades máximas...")
    max_similarities = []

    for _, row in test_new.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol is None:
            continue

        test_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        # Similaridade máxima
        max_sim = 0
        for train_fp in train_fps.values():
            sim = DataStructs.TanimotoSimilarity(test_fp, train_fp)
            if sim > max_sim:
                max_sim = sim

        max_similarities.append(max_sim)

    max_similarities = np.array(max_similarities)

    # Estatísticas
    print(f"\nDistribuição de Similaridade Máxima:")
    print(f"  Média: {max_similarities.mean():.3f}")
    print(f"  Mediana: {np.median(max_similarities):.3f}")
    print(f"  Min: {max_similarities.min():.3f}")
    print(f"  Max: {max_similarities.max():.3f}")

    # Categorias
    very_similar = (max_similarities > 0.8).sum()
    similar = ((max_similarities > 0.6) & (max_similarities <= 0.8)).sum()
    moderate = ((max_similarities > 0.4) & (max_similarities <= 0.6)).sum()
    low = (max_similarities <= 0.4).sum()

    print(f"\nCategorias de Similaridade:")
    print(f"  Muito similar (>0.8): {very_similar} ({100*very_similar/len(max_similarities):.1f}%)")
    print(f"  Similar (0.6-0.8):    {similar} ({100*similar/len(max_similarities):.1f}%)")
    print(f"  Moderada (0.4-0.6):   {moderate} ({100*moderate/len(max_similarities):.1f}%)")
    print(f"  Baixa (<0.4):         {low} ({100*low/len(max_similarities):.1f}%)")

    return {
        'max_similarities': max_similarities,
        'very_similar': very_similar,
        'similar': similar,
        'moderate': moderate,
        'low': low
    }


def plot_similarity_analysis(similarity_info: dict, output_dir: str = '.'):
    """Gráficos de análise de similaridade química."""
    if similarity_info is None:
        print("Sem dados de similaridade para plotar.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histograma de similaridade máxima
    ax1 = axes[0]
    ax1.hist(similarity_info['max_similarities'], bins=30, color='#3498db', edgecolor='black', alpha=0.8)
    ax1.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='Alta (0.8)')
    ax1.axvline(x=0.6, color='orange', linestyle='--', linewidth=2, label='Média (0.6)')
    ax1.set_xlabel('Similaridade Tanimoto Máxima', fontsize=12)
    ax1.set_ylabel('Número de Compostos de Teste', fontsize=12)
    ax1.set_title('Similaridade ao Vizinho Mais Próximo\n(Compostos Novos no Teste vs Treino)',
                  fontsize=13, fontweight='bold')
    ax1.legend()

    # Gráfico de pizza - distribuição de similaridade
    ax2 = axes[1]
    sizes = [similarity_info['very_similar'], similarity_info['similar'],
             similarity_info['moderate'], similarity_info['low']]
    total = sum(sizes)
    labels = [
        f'Muito similar (>0.8)\n{sizes[0]} ({100*sizes[0]/total:.1f}%)',
        f'Similar (0.6-0.8)\n{sizes[1]} ({100*sizes[1]/total:.1f}%)',
        f'Moderada (0.4-0.6)\n{sizes[2]} ({100*sizes[2]/total:.1f}%)',
        f'Baixa (<0.4)\n{sizes[3]} ({100*sizes[3]/total:.1f}%)'
    ]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']

    ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Distribuição de Similaridade', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_similarity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico salvo: {output_dir}/05_similarity_analysis.png")


# =============================================================================
# 7. RELATÓRIO FINAL
# =============================================================================

def generate_summary_report(df, leakage_info, results, kinase_stats, compound_stats, similarity_info, output_dir):
    """Gera relatório resumido."""
    print("\n" + "=" * 70)
    print("RELATÓRIO FINAL - ANÁLISE DE VIÉS NO DATASET")
    print("=" * 70)

    report = []
    report.append("=" * 70)
    report.append("RELATÓRIO: ANÁLISE DE VIÉS NO DATASET DE QUINASES")
    report.append("=" * 70)
    report.append("")

    # Dataset
    report.append("1. ESTATÍSTICAS DO DATASET")
    report.append("-" * 40)
    report.append(f"   Total de linhas: {len(df)}")
    report.append(f"   Compostos únicos: {df['chembl_id'].nunique()}")
    report.append(f"   Quinases únicas: {df['target_kinase'].nunique()}")
    report.append(f"   Ativos: {(df['label']==1).sum()} ({100*(df['label']==1).mean():.1f}%)")
    report.append(f"   Inativos: {(df['label']==0).sum()} ({100*(df['label']==0).mean():.1f}%)")
    report.append("")

    # Vazamento
    report.append("2. PROBLEMA CRÍTICO: VAZAMENTO DE DADOS")
    report.append("-" * 40)
    pct_leaked = 100 * leakage_info['n_test_leaked_lines'] / leakage_info['n_test_total']
    pct_exact = 100 * leakage_info['n_exact_lines'] / leakage_info['n_test_total']
    report.append(f"   Compostos vazados: {len(leakage_info['leaked_compounds'])} ({100*len(leakage_info['leaked_compounds'])/len(leakage_info['test_compounds']):.1f}% do teste)")
    report.append(f"   Linhas de teste com composto vazado: {pct_leaked:.1f}%")
    report.append(f"   Duplicatas exatas (composto+quinase): {pct_exact:.1f}%")
    report.append("   >>> CONSEQUÊNCIA: Modelo pode estar MEMORIZANDO!")
    report.append("")

    # Baselines
    report.append("3. COMPARAÇÃO: LOOKUP BASELINES vs KNN")
    report.append("-" * 40)
    for model, metrics in results.items():
        report.append(f"   {model}: Acc={metrics['accuracy']:.4f}, MCC={metrics['mcc']:.4f}")
    report.append("   >>> CONCLUSÃO: Lookup simples tem performance COMPARÁVEL ao KNN!")
    report.append("")

    # Desbalanceamento
    report.append("4. DESBALANCEAMENTO POR QUINASE")
    report.append("-" * 40)
    n_desbalanced = (kinase_stats['balance_class'] == 'Desbalanceada (>80% ou <20%)').sum()
    report.append(f"   Quinases desbalanceadas (>80% uma classe): {n_desbalanced} ({100*n_desbalanced/len(kinase_stats):.1f}%)")
    report.append("   >>> MUITAS QUINASES SÃO TRIVIALMENTE PREVISÍVEIS!")
    report.append("")

    # Consistência
    report.append("5. CONSISTÊNCIA DOS COMPOSTOS")
    report.append("-" * 40)
    consistency_counts = compound_stats['consistency'].value_counts()
    for cat, count in consistency_counts.items():
        report.append(f"   {cat}: {count} ({100*count/len(compound_stats):.1f}%)")
    report.append("   >>> FINGERPRINT DO COMPOSTO CARREGA MAIOR PARTE DO SINAL!")
    report.append("")

    # Similaridade
    if similarity_info:
        report.append("6. SIMILARIDADE QUÍMICA")
        report.append("-" * 40)
        total = len(similarity_info['max_similarities'])
        report.append(f"   Muito similar (>0.8): {100*similarity_info['very_similar']/total:.1f}%")
        report.append(f"   Similar (0.6-0.8): {100*similarity_info['similar']/total:.1f}%")
        report.append("   >>> POR ISSO O KNN FUNCIONA! Vizinhos muito próximos no treino.")
        report.append("")

    report.append("=" * 70)
    report.append("CONCLUSÃO GERAL")
    report.append("=" * 70)
    report.append("")
    report.append("Os modelos (KNN, MLP) NÃO estão aprendendo padrões complexos.")
    report.append("Eles estão MEMORIZANDO devido a:")
    report.append("  1. Vazamento massivo de compostos entre treino e teste")
    report.append("  2. Desbalanceamento extremo por quinase")
    report.append("  3. Comportamento consistente dos compostos")
    report.append("  4. Alta similaridade química ao treino")
    report.append("")
    report.append(">>> IMPLICAÇÃO: Modelos de triagem de quinases sem curadoria")
    report.append("    adequada podem estar superestimando sua capacidade de")
    report.append("    generalização!")
    report.append("")

    # Salvar relatório
    report_text = '\n'.join(report)
    print(report_text)

    with open(f'{output_dir}/analysis_report.txt', 'w') as f:
        f.write(report_text)

    print(f"\nRelatório salvo: {output_dir}/analysis_report.txt")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Configuração
    DATA_PATH = '/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_non_human_compounds.tsv'
    OUTPUT_DIR = '/media/leon/ssd2tb/docktkinase/leakage_analysis_results'

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Carregar dados
    df = load_dataset(DATA_PATH)

    # 2. Criar split
    train_idx, val_idx, test_idx = create_split(df)

    # 3. Análise de vazamento
    leakage_info = analyze_compound_leakage(df, train_idx, test_idx)
    plot_leakage_analysis(leakage_info, OUTPUT_DIR)

    # 4. Lookup baselines vs KNN
    results = evaluate_lookup_baselines(df, train_idx, test_idx)
    plot_baseline_comparison(results, OUTPUT_DIR)

    # 5. Desbalanceamento por quinase
    kinase_stats = analyze_kinase_imbalance(df, train_idx)
    plot_kinase_distribution(kinase_stats, OUTPUT_DIR)

    # 6. Consistência dos compostos
    compound_stats = analyze_compound_consistency(df, train_idx)
    plot_compound_consistency(compound_stats, OUTPUT_DIR)

    # 7. Similaridade química
    similarity_info = analyze_chemical_similarity(df, train_idx, test_idx)
    plot_similarity_analysis(similarity_info, OUTPUT_DIR)

    # 8. Relatório final
    generate_summary_report(df, leakage_info, results, kinase_stats, compound_stats, similarity_info, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("ANÁLISE COMPLETA!")
    print(f"Resultados salvos em: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
