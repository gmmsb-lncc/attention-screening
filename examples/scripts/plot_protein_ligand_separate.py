#!/usr/bin/env python3
"""
Gráfico de Comparação: Embeddings de Proteína vs Ligante
==========================================================

Cria gráficos separados comparando KNN e MLP usando:
1. Apenas embeddings de proteína
2. Apenas embeddings de ligante

Layout: 2 linhas x 3 colunas
- Linha 1: MCC para 8M, 150M, 3B
- Linha 2: AUC-ROC para 8M, 150M, 3B

Cada subplot mostra KNN vs MLP com barras de erro.

Autor: DockTKinase Project
Data: Janeiro 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from tqdm import tqdm

# Configurações
EMBEDDINGS_DIR = Path('/data/docktkinase/results/protein_model_benchmark_human_v2')
OUTPUT_DIR = Path('results/baseline_multiseed_human')
DATASET_PATH = Path('tests/datasets/kinase_human_compounds.tsv')
BASELINE_RESULTS_FILE = Path('results/baseline_multiseed_human/baseline_multiseed_results.json')

# Seeds para reprodutibilidade
SEEDS = [42, 123, 420, 777, 2024]

# Modelos
MODELS = {
    'esm2_t6_8M_UR50D': '8M',
    'esm2_t30_150M_UR50D': '150M',
    'esm2_t36_3B_UR50D': '3B',
}

# Cores
COLORS = {
    'KNN': '#3498db',  # Azul
    'MLP': '#e74c3c',  # Vermelho
}


def load_protein_embeddings(model_name):
    """Carrega embeddings apenas de proteínas."""
    import pandas as pd
    
    # Carregar dataset para mapear seq_id -> chembl_id
    df = pd.read_csv(DATASET_PATH, sep='\t')
    
    # Carregar embeddings de proteínas
    protein_dir = EMBEDDINGS_DIR / model_name / 'build' / 'proteins'
    labels_path = EMBEDDINGS_DIR / model_name / 'build' / 'binary_labels.npy'
    
    labels = np.load(labels_path, allow_pickle=True)
    
    # Criar dicionário de embeddings por seq_id
    protein_embeddings = {}
    for seq_id in df['seq_id'].unique():
        emb_path = protein_dir / f"{seq_id}_embedding.npy"
        if emb_path.exists():
            protein_embeddings[seq_id] = np.load(emb_path)
    
    # Montar matriz de embeddings na ordem do dataset
    embeddings_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        seq_id = row['seq_id']
        if seq_id in protein_embeddings:
            embeddings_list.append(protein_embeddings[seq_id])
            valid_indices.append(idx)
    
    embeddings = np.vstack(embeddings_list)
    labels = labels[valid_indices]
    
    # Filtrar labels válidos
    valid_mask = np.isin(labels, [0, 1])
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask].astype(int)
    
    return embeddings, labels


def load_ligand_embeddings(model_name):
    """Carrega embeddings apenas de ligantes."""
    import pandas as pd
    
    # Carregar dataset
    df = pd.read_csv(DATASET_PATH, sep='\t')
    
    # Carregar embeddings de ligantes
    ligand_dir = EMBEDDINGS_DIR / model_name / 'build' / 'ligand_matrices'
    labels_path = EMBEDDINGS_DIR / model_name / 'build' / 'binary_labels.npy'
    
    labels = np.load(labels_path, allow_pickle=True)
    
    # Criar dicionário de embeddings por chembl_id
    ligand_embeddings = {}
    for chembl_id in df['chembl_id'].unique():
        emb_path = ligand_dir / f"{chembl_id}_matrix.npy"
        if emb_path.exists():
            emb = np.load(emb_path)
            # Ligante é (1, 768) - pegar vetor 1D
            if emb.ndim == 2:
                emb = emb[0]
            ligand_embeddings[chembl_id] = emb
    
    # Montar matriz de embeddings na ordem do dataset
    embeddings_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        chembl_id = row['chembl_id']
        if chembl_id in ligand_embeddings:
            embeddings_list.append(ligand_embeddings[chembl_id])
            valid_indices.append(idx)
    
    embeddings = np.vstack(embeddings_list)
    labels = labels[valid_indices]
    
    # Filtrar labels válidos
    valid_mask = np.isin(labels, [0, 1])
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask].astype(int)
    
    return embeddings, labels


def evaluate_model(X_train, X_test, y_train, y_test, classifier_type='KNN'):
    """Treina e avalia um modelo."""
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Criar modelo
    if classifier_type == 'KNN':
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine'
        )
    else:  # MLP
        model = MLPClassifier(
            hidden_layer_sizes=(512,),
            max_iter=500,
            random_state=SEEDS[0],
            early_stopping=True
        )
    
    # Treinar
    model.fit(X_train_scaled, y_train)
    
    # Predizer
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # Métricas
    auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    return auc, mcc


def run_multiseed_evaluation(embeddings, labels, embedding_type='protein'):
    """Executa avaliação com múltiplas seeds."""
    results = {
        'KNN': {'auc': [], 'mcc': []},
        'MLP': {'auc': [], 'mcc': []},
    }
    
    print(f'\n🧬 Avaliando {embedding_type.upper()} embeddings...')
    
    for seed in tqdm(SEEDS, desc=f'  Seeds'):
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels,
            test_size=0.1,
            random_state=seed,
            stratify=labels
        )
        
        # Avaliar KNN
        auc_knn, mcc_knn = evaluate_model(X_train, X_test, y_train, y_test, 'KNN')
        results['KNN']['auc'].append(auc_knn)
        results['KNN']['mcc'].append(mcc_knn)
        
        # Avaliar MLP
        auc_mlp, mcc_mlp = evaluate_model(X_train, X_test, y_train, y_test, 'MLP')
        results['MLP']['auc'].append(auc_mlp)
        results['MLP']['mcc'].append(mcc_mlp)
    
    # Calcular estatísticas
    stats = {}
    for clf in ['KNN', 'MLP']:
        stats[clf] = {
            'auc': {
                'mean': np.mean(results[clf]['auc']),
                'std': np.std(results[clf]['auc']),
            },
            'mcc': {
                'mean': np.mean(results[clf]['mcc']),
                'std': np.std(results[clf]['mcc']),
            }
        }
    
    return stats


def load_combined_stats():
    """Carrega estatísticas de proteína+ligante concatenado."""
    if not BASELINE_RESULTS_FILE.exists():
        return None
    
    with open(BASELINE_RESULTS_FILE, 'r') as f:
        data = json.load(f)
    
    # Extrair estatísticas agregadas
    combined_stats = {}
    for model_name in MODELS.keys():
        if model_name in data['models']:
            model_data = data['models'][model_name]['aggregated']
            combined_stats[model_name] = {
                'KNN': {
                    'auc': model_data['KNN']['roc_auc'],
                    'mcc': model_data['KNN']['mcc']
                },
                'MLP': {
                    'auc': model_data['MLP']['roc_auc'],
                    'mcc': model_data['MLP']['mcc']
                }
            }
    
    return combined_stats


def create_comparison_plot(protein_stats, ligand_stats, combined_stats=None):
    """Cria gráfico de comparação 2x3 (MCC e AUC-ROC)."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    model_names = list(MODELS.keys())
    model_labels = list(MODELS.values())
    
    width = 0.22  # Reduzido para acomodar 3 barras
    x = np.array([0, 1])  # KNN, MLP
    
    # Para cada modelo
    for col_idx, model_name in enumerate(model_names):
        model_label = model_labels[col_idx]
        
        # ========== LINHA 1: MCC ==========
        ax_mcc = axes[0, col_idx]
        
        # Dados de proteína
        protein_mcc_means = [
            protein_stats[model_name]['KNN']['mcc']['mean'],
            protein_stats[model_name]['MLP']['mcc']['mean']
        ]
        protein_mcc_stds = [
            protein_stats[model_name]['KNN']['mcc']['std'],
            protein_stats[model_name]['MLP']['mcc']['std']
        ]
        
        # Dados de ligante
        ligand_mcc_means = [
            ligand_stats[model_name]['KNN']['mcc']['mean'],
            ligand_stats[model_name]['MLP']['mcc']['mean']
        ]
        ligand_mcc_stds = [
            ligand_stats[model_name]['KNN']['mcc']['std'],
            ligand_stats[model_name]['MLP']['mcc']['std']
        ]
        
        # Dados de proteína+ligante (se disponível)
        if combined_stats:
            combined_mcc_means = [
                combined_stats[model_name]['KNN']['mcc']['mean'],
                combined_stats[model_name]['MLP']['mcc']['mean']
            ]
            combined_mcc_stds = [
                combined_stats[model_name]['KNN']['mcc']['std'],
                combined_stats[model_name]['MLP']['mcc']['std']
            ]
        
        # Barras
        bars_prot = ax_mcc.bar(
            x - width, protein_mcc_means, width,
            yerr=protein_mcc_stds, label='Proteína',
            color='#5B9BD5', alpha=1.0, capsize=5, edgecolor='black', linewidth=1
        )
        bars_lig = ax_mcc.bar(
            x, ligand_mcc_means, width,
            yerr=ligand_mcc_stds, label='Ligante',
            color='#ED7D31', alpha=1.0, capsize=5, edgecolor='black', linewidth=1
        )
        
        if combined_stats:
            bars_comb = ax_mcc.bar(
                x + width, combined_mcc_means, width,
                yerr=combined_mcc_stds, label='Prot+Lig',
                color='#70AD47', alpha=1.0, capsize=5, edgecolor='black', linewidth=1
            )
        
        # Não mostrar valores numéricos (removido para reduzir poluição visual)
        
        # Configurações
        ax_mcc.set_ylabel('MCC', fontsize=12, fontweight='bold')
        ax_mcc.set_title(f'{model_label}', fontsize=14, fontweight='bold')
        ax_mcc.set_xticks(x)
        ax_mcc.set_xticklabels(['KNN', 'MLP'], fontsize=11)
        ax_mcc.set_ylim(0, 1.0)
        ax_mcc.grid(axis='y', alpha=0.3, linestyle='--')
        
        # ========== LINHA 2: AUC-ROC ==========
        ax_auc = axes[1, col_idx]
        
        # Dados de proteína
        protein_auc_means = [
            protein_stats[model_name]['KNN']['auc']['mean'],
            protein_stats[model_name]['MLP']['auc']['mean']
        ]
        protein_auc_stds = [
            protein_stats[model_name]['KNN']['auc']['std'],
            protein_stats[model_name]['MLP']['auc']['std']
        ]
        
        # Dados de ligante
        ligand_auc_means = [
            ligand_stats[model_name]['KNN']['auc']['mean'],
            ligand_stats[model_name]['MLP']['auc']['mean']
        ]
        ligand_auc_stds = [
            ligand_stats[model_name]['KNN']['auc']['std'],
            ligand_stats[model_name]['MLP']['auc']['std']
        ]
        
        # Dados de proteína+ligante (se disponível)
        if combined_stats:
            combined_auc_means = [
                combined_stats[model_name]['KNN']['auc']['mean'],
                combined_stats[model_name]['MLP']['auc']['mean']
            ]
            combined_auc_stds = [
                combined_stats[model_name]['KNN']['auc']['std'],
                combined_stats[model_name]['MLP']['auc']['std']
            ]
        
        # Barras
        bars_prot = ax_auc.bar(
            x - width, protein_auc_means, width,
            yerr=protein_auc_stds, label='Proteína',
            color='#5B9BD5', alpha=1.0, capsize=5, edgecolor='black', linewidth=1
        )
        bars_lig = ax_auc.bar(
            x, ligand_auc_means, width,
            yerr=ligand_auc_stds, label='Ligante',
            color='#ED7D31', alpha=1.0, capsize=5, edgecolor='black', linewidth=1
        )
        
        if combined_stats:
            bars_comb = ax_auc.bar(
                x + width, combined_auc_means, width,
                yerr=combined_auc_stds, label='Prot+Lig',
                color='#70AD47', alpha=1.0, capsize=5, edgecolor='black', linewidth=1
            )
        
        # Não mostrar valores numéricos (removido para reduzir poluição visual)
        
        # Configurações
        ax_auc.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
        ax_auc.set_xlabel('Classificador', fontsize=12, fontweight='bold')
        ax_auc.set_xticks(x)
        ax_auc.set_xticklabels(['KNN', 'MLP'], fontsize=11)
        ax_auc.set_ylim(0, 1.0)
        ax_auc.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Remover legendas individuais dos subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.legend().set_visible(False)
    
    # Criar legenda única na figura (fora dos gráficos)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper right', 
               bbox_to_anchor=(1.02, 1.0),
               fontsize=11, 
               framealpha=0.95,
               edgecolor='black',
               fancybox=False)
    
    # Título geral
    fig.suptitle(
        'Comparação: Proteína vs Ligante vs Proteína+Ligante\n(5 seeds, split aleatório)',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    
    # Salvar
    output_file = OUTPUT_DIR / 'protein_vs_ligand_comparison.png'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\n✅ Gráfico salvo em: {output_file}')
    
    plt.close()


def main():
    """Executa análise completa."""
    print('=' * 70)
    print('📊 ANÁLISE: PROTEÍNA vs LIGANTE')
    print('=' * 70)
    
    # Verificar se resultados já existem
    stats_file = OUTPUT_DIR / 'protein_vs_ligand_stats.json'
    
    if stats_file.exists():
        print('\n✅ Carregando resultados existentes...')
        with open(stats_file, 'r') as f:
            data = json.load(f)
            protein_stats = data['protein']
            ligand_stats = data['ligand']
        print('   Dados carregados com sucesso!')
        
        # Carregar resultados de proteína+ligante
        print('   Carregando resultados proteína+ligante...')
        combined_stats = load_combined_stats()
        if combined_stats:
            print('   ✅ Dados proteína+ligante carregados!')
        else:
            print('   ⚠️  Arquivo baseline_multiseed_results.json não encontrado')
        
        # Mostrar resumo
        for model_name, model_label in MODELS.items():
            if model_name in protein_stats:
                print(f'\n   📊 {model_label}:')
                print(f'   PROTEÍNA     - KNN: AUC={protein_stats[model_name]["KNN"]["auc"]["mean"]:.4f}, MCC={protein_stats[model_name]["KNN"]["mcc"]["mean"]:.4f}')
                print(f'   PROTEÍNA     - MLP: AUC={protein_stats[model_name]["MLP"]["auc"]["mean"]:.4f}, MCC={protein_stats[model_name]["MLP"]["mcc"]["mean"]:.4f}')
                print(f'   LIGANTE      - KNN: AUC={ligand_stats[model_name]["KNN"]["auc"]["mean"]:.4f}, MCC={ligand_stats[model_name]["KNN"]["mcc"]["mean"]:.4f}')
                print(f'   LIGANTE      - MLP: AUC={ligand_stats[model_name]["MLP"]["auc"]["mean"]:.4f}, MCC={ligand_stats[model_name]["MLP"]["mcc"]["mean"]:.4f}')
                if combined_stats and model_name in combined_stats:
                    print(f'   PROT+LIGANTE - KNN: AUC={combined_stats[model_name]["KNN"]["auc"]["mean"]:.4f}, MCC={combined_stats[model_name]["KNN"]["mcc"]["mean"]:.4f}')
                    print(f'   PROT+LIGANTE - MLP: AUC={combined_stats[model_name]["MLP"]["auc"]["mean"]:.4f}, MCC={combined_stats[model_name]["MLP"]["mcc"]["mean"]:.4f}')
    else:
        print('\n⏳ Calculando resultados (isso será feito apenas uma vez)...')
        
        protein_stats = {}
        ligand_stats = {}
        combined_stats = None
        
        for model_name, model_label in MODELS.items():
            print(f'\n📂 Processando modelo: {model_label}')
            print('-' * 70)
            
            # Carregar embeddings de proteína
            print('   Carregando embeddings de proteína...')
            protein_emb, protein_labels = load_protein_embeddings(model_name)
            print(f'   Proteína: {protein_emb.shape}')
            
            # Carregar embeddings de ligante
            print('   Carregando embeddings de ligante...')
            ligand_emb, ligand_labels = load_ligand_embeddings(model_name)
            print(f'   Ligante: {ligand_emb.shape}')
            
            # Avaliar proteína
            protein_stats[model_name] = run_multiseed_evaluation(
                protein_emb, protein_labels, 'protein'
            )
            
            # Avaliar ligante
            ligand_stats[model_name] = run_multiseed_evaluation(
                ligand_emb, ligand_labels, 'ligand'
            )
            
            # Mostrar resultados
            print(f'\n   📊 Resultados {model_label}:')
            print(f'   PROTEÍNA - KNN: AUC={protein_stats[model_name]["KNN"]["auc"]["mean"]:.4f}, MCC={protein_stats[model_name]["KNN"]["mcc"]["mean"]:.4f}')
            print(f'   PROTEÍNA - MLP: AUC={protein_stats[model_name]["MLP"]["auc"]["mean"]:.4f}, MCC={protein_stats[model_name]["MLP"]["mcc"]["mean"]:.4f}')
            print(f'   LIGANTE  - KNN: AUC={ligand_stats[model_name]["KNN"]["auc"]["mean"]:.4f}, MCC={ligand_stats[model_name]["KNN"]["mcc"]["mean"]:.4f}')
            print(f'   LIGANTE  - MLP: AUC={ligand_stats[model_name]["MLP"]["auc"]["mean"]:.4f}, MCC={ligand_stats[model_name]["MLP"]["mcc"]["mean"]:.4f}')
        
        # Carregar resultados de proteína+ligante
        print('\n   Carregando resultados proteína+ligante...')
        combined_stats = load_combined_stats()
        if combined_stats:
            print('   ✅ Dados proteína+ligante carregados!')
        else:
            print('   ⚠️  Arquivo baseline_multiseed_results.json não encontrado')
        
        # Salvar estatísticas
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump({
                'protein': protein_stats,
                'ligand': ligand_stats
            }, f, indent=2)
        print(f'\n💾 Estatísticas salvas em: {stats_file}')
    
    # Criar gráfico (sempre gera novo gráfico)
    print('\n' + '=' * 70)
    print('📈 GERANDO GRÁFICO...')
    print('=' * 70)
    create_comparison_plot(protein_stats, ligand_stats, combined_stats)
    
    print('\n' + '=' * 70)
    print('✅ CONCLUÍDO')
    print('=' * 70)


if __name__ == '__main__':
    main()
