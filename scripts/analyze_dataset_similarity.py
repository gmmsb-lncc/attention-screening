#!/usr/bin/env python3
"""
Análise Completa de Similaridade de Dataset
============================================

Script centralizado que executa todas as análises de similaridade:
1. Similaridade geral train-test
2. Similaridade por classe (proteína e ligante separados)
3. Comparação proteína vs ligante vs combinado
4. Visualizações alternativas (boxplots, heatmaps)

Gera os principais gráficos:
- protein_vs_ligand_comparison.png
- class_similarity_combined.png
- Boxplots e heatmaps

Uso:
    python analyze_dataset_similarity.py --embeddings-dir /path/to/embeddings --output-dir results/analysis

Autor: DockTKinase Project
Data: Janeiro 2026
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from tqdm import tqdm


class DatasetSimilarityAnalyzer:
    """Analisador de similaridade de datasets."""
    
    def __init__(self, embeddings_dir, output_dir, dataset_name='Dataset', 
                 random_seed=42, test_size=0.1, val_size=0.1):
        """
        Inicializa o analisador.
        
        Args:
            embeddings_dir: Diretório com embeddings
            output_dir: Diretório de saída
            dataset_name: Nome do dataset (para títulos)
            random_seed: Seed aleatória
            test_size: Proporção do conjunto de teste
            val_size: Proporção do conjunto de validação
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.random_seed = random_seed
        self.test_size = test_size
        self.val_size = val_size
        
        # Descobrir modelos automaticamente
        self.models = self._discover_models()
        
        # Armazenar resultados
        self.results = {}
        
    def _discover_models(self):
        """Descobre modelos disponíveis no diretório."""
        # Modelos desejados (apenas 8M, 150M e 3B)
        target_models = {
            'esm2_t6_8M_UR50D': '8M',
            'esm2_t30_150M_UR50D': '150M',
            'esm2_t36_3B_UR50D': '3B',
        }
        
        models = {}
        if not self.embeddings_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.embeddings_dir}")
        
        for model_dir in self.embeddings_dir.iterdir():
            if model_dir.is_dir() and (model_dir / 'build').exists():
                model_name = model_dir.name
                
                # Apenas processar modelos específicos
                if model_name in target_models:
                    models[model_name] = target_models[model_name]
        
        if not models:
            raise ValueError(f"Nenhum dos modelos esperados (8M, 150M, 3B) encontrado em {self.embeddings_dir}")
        
        print(f"✅ Modelos encontrados: {list(models.keys())}")
        return models
    
    def load_embeddings(self, model_name):
        """Carrega embeddings e labels."""
        build_dir = self.embeddings_dir / model_name / 'build'
        
        embeddings = np.load(build_dir / 'embedding_matrix.npy', allow_pickle=True)
        labels = np.load(build_dir / 'binary_labels.npy', allow_pickle=True)
        
        # Garantir formato correto
        if embeddings.ndim == 1:
            embeddings = np.vstack(embeddings)
        if labels.ndim > 1:
            labels = labels.ravel()
        
        # Filtrar válidos
        valid_mask = np.isin(labels, [0, 1])
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask].astype(int)
        
        return embeddings, labels
    
    def split_data(self, embeddings, labels):
        """Realiza split train/test."""
        X_train, X_temp, y_train, y_temp = train_test_split(
            embeddings, labels,
            test_size=self.test_size + self.val_size,
            random_state=self.random_seed,
            stratify=labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.test_size / (self.test_size + self.val_size),
            random_state=self.random_seed,
            stratify=y_temp
        )
        
        return X_train, X_test, y_train, y_test
    
    def calculate_class_similarities(self, X_train, y_train, X_test, y_test):
        """Calcula similaridades entre classes."""
        # Separar treino por classe
        X_train_pos = X_train[y_train == 1]
        X_train_neg = X_train[y_train == 0]
        
        # Separar teste por classe
        X_test_pos = X_test[y_test == 1]
        X_test_neg = X_test[y_test == 0]
        
        results = {}
        
        # POS → POS
        sim_matrix = cosine_similarity(X_test_pos, X_train_pos)
        results['train_pos_test_pos'] = sim_matrix.max(axis=1)
        
        # NEG → POS
        sim_matrix = cosine_similarity(X_test_pos, X_train_neg)
        results['train_neg_test_pos'] = sim_matrix.max(axis=1)
        
        # POS → NEG
        sim_matrix = cosine_similarity(X_test_neg, X_train_pos)
        results['train_pos_test_neg'] = sim_matrix.max(axis=1)
        
        # NEG → NEG
        sim_matrix = cosine_similarity(X_test_neg, X_train_neg)
        results['train_neg_test_neg'] = sim_matrix.max(axis=1)
        
        return results
    
    def separate_protein_ligand(self, embeddings, ligand_dim=768):
        """Separa embeddings em proteína e ligante."""
        protein_dim = embeddings.shape[1] - ligand_dim
        protein_emb = embeddings[:, :protein_dim]
        ligand_emb = embeddings[:, protein_dim:]
        return protein_emb, ligand_emb
    
    def evaluate_classifier(self, X_train, y_train, X_test, y_test, classifier_type='knn'):
        """Avalia classificador."""
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Criar classificador
        if classifier_type == 'knn':
            clf = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine')
        else:  # mlp
            clf = MLPClassifier(
                hidden_layer_sizes=(512,),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.random_seed,
                verbose=False
            )
        
        # Treinar
        clf.fit(X_train_scaled, y_train)
        
        # Predizer
        if hasattr(clf, 'predict_proba'):
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = clf.predict(X_test_scaled)
        
        y_pred = clf.predict(X_test_scaled)
        
        # Métricas
        auc = roc_auc_score(y_test, y_pred_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        return auc, mcc
    
    def analyze_component_performance(self):
        """Analisa performance por componente (proteína vs ligante)."""
        print('\n' + '='*70)
        print('📊 ANÁLISE DE PERFORMANCE POR COMPONENTE')
        print('='*70)
        
        component_results = {'protein': {}, 'ligand': {}, 'combined': {}}
        
        for model_name, label in self.models.items():
            print(f'\n🧬 Processando modelo {label}...')
            
            # Carregar dados
            embeddings, labels = self.load_embeddings(model_name)
            protein_emb, ligand_emb = self.separate_protein_ligand(embeddings)
            
            # Split
            X_train, X_test, y_train, y_test = self.split_data(embeddings, labels)
            X_train_prot, X_train_lig = self.separate_protein_ligand(X_train)
            X_test_prot, X_test_lig = self.separate_protein_ligand(X_test)
            
            # Avaliar cada componente
            for component_name, X_tr, X_te in [
                ('protein', X_train_prot, X_test_prot),
                ('ligand', X_train_lig, X_test_lig),
                ('combined', X_train, X_test)
            ]:
                knn_auc, knn_mcc = self.evaluate_classifier(X_tr, y_train, X_te, y_test, 'knn')
                mlp_auc, mlp_mcc = self.evaluate_classifier(X_tr, y_train, X_te, y_test, 'mlp')
                
                component_results[component_name][model_name] = {
                    'knn': {'auc': knn_auc, 'mcc': knn_mcc},
                    'mlp': {'auc': mlp_auc, 'mcc': mlp_mcc}
                }
                
                print(f'   {component_name:10s}: KNN AUC={knn_auc:.3f} | MLP AUC={mlp_auc:.3f}')
        
        self.results['component_performance'] = component_results
        return component_results
    
    def analyze_class_similarity(self):
        """Analisa similaridade por classe."""
        print('\n' + '='*70)
        print('📊 ANÁLISE DE SIMILARIDADE POR CLASSE')
        print('='*70)
        
        protein_results = {}
        ligand_results = {}
        
        for model_name, label in self.models.items():
            print(f'\n🧬 Processando modelo {label}...')
            
            # Carregar dados
            embeddings, labels = self.load_embeddings(model_name)
            protein_emb, ligand_emb = self.separate_protein_ligand(embeddings)
            
            # Split
            X_train, X_test, y_train, y_test = self.split_data(embeddings, labels)
            X_train_prot, X_train_lig = self.separate_protein_ligand(X_train)
            X_test_prot, X_test_lig = self.separate_protein_ligand(X_test)
            
            # Similaridades - Proteína
            print('   Calculando similaridades - Proteína...')
            protein_sim = self.calculate_class_similarities(X_train_prot, y_train, X_test_prot, y_test)
            protein_results[model_name] = protein_sim
            
            # Similaridades - Ligante
            print('   Calculando similaridades - Ligante...')
            ligand_sim = self.calculate_class_similarities(X_train_lig, y_train, X_test_lig, y_test)
            ligand_results[model_name] = ligand_sim
            
            # Separabilidade
            prot_sep = (protein_sim['train_pos_test_pos'].mean() + protein_sim['train_neg_test_neg'].mean()) / 2 - \
                       (protein_sim['train_pos_test_neg'].mean() + protein_sim['train_neg_test_pos'].mean()) / 2
            lig_sep = (ligand_sim['train_pos_test_pos'].mean() + ligand_sim['train_neg_test_neg'].mean()) / 2 - \
                      (ligand_sim['train_pos_test_neg'].mean() + ligand_sim['train_neg_test_pos'].mean()) / 2
            
            print(f'   Separabilidade Proteína: {prot_sep:.4f}')
            print(f'   Separabilidade Ligante: {lig_sep:.4f}')
        
        self.results['class_similarity'] = {
            'protein': protein_results,
            'ligand': ligand_results
        }
        
        return protein_results, ligand_results
    
    def plot_protein_vs_ligand(self, component_results):
        """Gera gráfico de comparação proteína vs ligante."""
        print('\n📊 Gerando gráfico: protein_vs_ligand_comparison.png')
        
        fig, axes = plt.subplots(2, len(self.models), figsize=(5*len(self.models), 8))
        if len(self.models) == 1:
            axes = axes.reshape(2, 1)
        
        components = ['protein', 'ligand', 'combined']
        colors = ['#5B9BD5', '#ED7D31', '#70AD47']
        
        for col, (model_name, label) in enumerate(self.models.items()):
            # MCC (top)
            ax_mcc = axes[0, col]
            knn_mccs = [component_results[comp][model_name]['knn']['mcc'] for comp in components]
            mlp_mccs = [component_results[comp][model_name]['mlp']['mcc'] for comp in components]
            
            x = np.arange(len(components))
            width = 0.35
            
            for i, (mccs, clf_label) in enumerate([(knn_mccs, 'KNN'), (mlp_mccs, 'MLP')]):
                offset = (i - 0.5) * width
                bars = ax_mcc.bar(x + offset, mccs, width, label=clf_label)
                for j, bar in enumerate(bars):
                    bar.set_color(colors[j])
                    bar.set_alpha(0.7)
                    bar.set_edgecolor('black')
            
            ax_mcc.set_ylabel('MCC', fontsize=11, fontweight='bold')
            ax_mcc.set_title(f'Modelo {label}', fontsize=12, fontweight='bold')
            ax_mcc.set_xticks(x)
            ax_mcc.set_xticklabels(['Proteína', 'Ligante', 'Combinado'])
            ax_mcc.legend()
            ax_mcc.grid(True, alpha=0.3, axis='y')
            ax_mcc.set_ylim(0, 1)
            
            # AUC-ROC (bottom)
            ax_auc = axes[1, col]
            knn_aucs = [component_results[comp][model_name]['knn']['auc'] for comp in components]
            mlp_aucs = [component_results[comp][model_name]['mlp']['auc'] for comp in components]
            
            for i, (aucs, clf_label) in enumerate([(knn_aucs, 'KNN'), (mlp_aucs, 'MLP')]):
                offset = (i - 0.5) * width
                bars = ax_auc.bar(x + offset, aucs, width, label=clf_label)
                for j, bar in enumerate(bars):
                    bar.set_color(colors[j])
                    bar.set_alpha(0.7)
                    bar.set_edgecolor('black')
            
            ax_auc.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
            ax_auc.set_xticks(x)
            ax_auc.set_xticklabels(['Proteína', 'Ligante', 'Combinado'])
            ax_auc.legend()
            ax_auc.grid(True, alpha=0.3, axis='y')
            ax_auc.set_ylim(0, 1)
        
        fig.suptitle(f'Comparação de Performance: Proteína vs Ligante vs Combinado ({self.dataset_name})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'protein_vs_ligand_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'   ✅ Salvo: {output_file}')
        plt.close()
    
    def plot_class_similarity_combined(self, protein_results, ligand_results):
        """Gera gráfico combinado de similaridade por classe."""
        print('\n📊 Gerando gráfico: class_similarity_combined.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        scenarios = [
            ('train_pos_test_pos', 'Treino POS → Teste POS'),
            ('train_neg_test_pos', 'Treino NEG → Teste POS'),
            ('train_pos_test_neg', 'Treino POS → Teste NEG'),
            ('train_neg_test_neg', 'Treino NEG → Teste NEG'),
        ]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))
        
        for idx, (key, title) in enumerate(scenarios):
            ax = axes[idx // 2, idx % 2]
            
            # Plotar proteína
            for i, (model_name, label) in enumerate(self.models.items()):
                data = protein_results[model_name][key]
                ax.hist(data, bins=50, alpha=0.5, label=f'{label} Prot', 
                       color=colors[i], edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Similaridade de Cosseno', fontsize=10)
            ax.set_ylabel('Frequência', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 1.0)
        
        fig.suptitle(f'Similaridade por Classe - Proteína ({self.dataset_name})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'class_similarity_combined.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'   ✅ Salvo: {output_file}')
        plt.close()
    
    def run_full_analysis(self):
        """Executa análise completa."""
        print('\n' + '='*70)
        print(f'🔬 ANÁLISE COMPLETA DE SIMILARIDADE - {self.dataset_name}')
        print('='*70)
        print(f'📁 Embeddings: {self.embeddings_dir}')
        print(f'📁 Output: {self.output_dir}')
        print('='*70)
        
        # Criar diretório de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Análise de performance por componente
        component_results = self.analyze_component_performance()
        
        # 2. Análise de similaridade por classe
        protein_results, ligand_results = self.analyze_class_similarity()
        
        # 3. Gerar visualizações
        print('\n' + '='*70)
        print('📊 GERANDO VISUALIZAÇÕES')
        print('='*70)
        
        self.plot_protein_vs_ligand(component_results)
        self.plot_class_similarity_combined(protein_results, ligand_results)
        
        # 4. Salvar resultados
        results_file = self.output_dir / 'similarity_analysis_results.json'
        
        # Converter arrays para listas para JSON
        json_results = {}
        for comp_name, comp_data in component_results.items():
            json_results[comp_name] = comp_data
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f'\n💾 Resultados salvos: {results_file}')
        
        print('\n' + '='*70)
        print('✅ ANÁLISE COMPLETA CONCLUÍDA!')
        print('='*70)
        print(f'\n📁 Arquivos gerados em: {self.output_dir}')
        print('   - protein_vs_ligand_comparison.png')
        print('   - class_similarity_combined.png')
        print('   - similarity_analysis_results.json')


def main():
    parser = argparse.ArgumentParser(
        description='Análise completa de similaridade de dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='/data/docktkinase/results/protein_model_benchmark_human_v2',
        help='Diretório com embeddings (padrão: human_v2)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/similarity_analysis_human',
        help='Diretório de saída (padrão: results/similarity_analysis_human)'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='HUMAN',
        help='Nome do dataset para títulos (padrão: HUMAN)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed aleatória (padrão: 42)'
    )
    
    args = parser.parse_args()
    
    # Criar analisador e executar
    analyzer = DatasetSimilarityAnalyzer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        random_seed=args.seed
    )
    
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
