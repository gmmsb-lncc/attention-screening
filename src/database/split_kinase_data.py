#!/usr/bin/env python3
"""
Script para separar dados de kinases em 3 arquivos:
1. kinase_all_compounds.tsv (todos - já existe)
2. kinase_human_compounds.tsv (apenas Homo sapiens)
3. kinase_non_human_compounds.tsv (exceto Homo sapiens)
"""

import pandas as pd
import os
from pathlib import Path

def split_kinase_data(input_file, output_dir):
    """
    Separa os dados de kinases em humanos e não-humanos.
    
    Args:
        input_file: Caminho para o arquivo kinase_all_compounds.tsv
        output_dir: Diretório onde os arquivos serão salvos
    """
    
    print("=" * 80)
    print("🧬 SEPARAÇÃO DE DADOS DE KINASES")
    print("=" * 80)
    
    # Verificar se o arquivo existe
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {input_file}")
    
    print(f"\n📂 Lendo arquivo: {input_file}")
    print(f"📊 Tamanho do arquivo: {os.path.getsize(input_file) / (1024**3):.2f} GB")
    
    # Ler o arquivo TSV
    print("\n⏳ Carregando dados (pode levar alguns minutos)...")
    df = pd.read_csv(input_file, sep='\t', low_memory=False)
    
    print(f"✅ Dados carregados: {len(df):,} registros")
    print(f"\n📋 Colunas disponíveis:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Verificar valores únicos na coluna organism
    print(f"\n🌍 Organismos únicos encontrados: {df['organism'].nunique():,}")
    print("\n📊 Top 10 organismos mais frequentes:")
    organism_counts = df['organism'].value_counts()
    for org, count in organism_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"  • {org}: {count:,} ({percentage:.2f}%)")
    
    # Separar dados
    print("\n" + "=" * 80)
    print("🔬 SEPARANDO DADOS...")
    print("=" * 80)
    
    # Filtrar humanos
    df_humans = df[df['organism'] == 'Homo sapiens'].copy()
    print(f"\n✅ Dados humanos: {len(df_humans):,} registros ({len(df_humans)/len(df)*100:.2f}%)")
    
    # Filtrar não-humanos
    df_non_humans = df[df['organism'] != 'Homo sapiens'].copy()
    print(f"✅ Dados não-humanos: {len(df_non_humans):,} registros ({len(df_non_humans)/len(df)*100:.2f}%)")
    
    # Verificar soma
    assert len(df_humans) + len(df_non_humans) == len(df), "❌ Erro: soma não confere!"
    print(f"✅ Verificação: {len(df_humans):,} + {len(df_non_humans):,} = {len(df):,} ✓")
    
    # Criar diretório de saída se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Salvar arquivos
    print("\n" + "=" * 80)
    print("💾 SALVANDO ARQUIVOS...")
    print("=" * 80)
    
    # Arquivo de humanos
    output_humans = os.path.join(output_dir, 'kinase_human_compounds.tsv')
    print(f"\n📝 Salvando: {output_humans}")
    df_humans.to_csv(output_humans, sep='\t', index=False)
    size_humans = os.path.getsize(output_humans) / (1024**2)
    print(f"   ✅ Salvo: {size_humans:.2f} MB")
    
    # Arquivo de não-humanos
    output_non_humans = os.path.join(output_dir, 'kinase_non_human_compounds.tsv')
    print(f"\n📝 Salvando: {output_non_humans}")
    df_non_humans.to_csv(output_non_humans, sep='\t', index=False)
    size_non_humans = os.path.getsize(output_non_humans) / (1024**2)
    print(f"   ✅ Salvo: {size_non_humans:.2f} MB")
    
    # Copiar arquivo original para o diretório de saída (se necessário)
    output_all = os.path.join(output_dir, 'kinase_all_compounds.tsv')
    if os.path.abspath(input_file) != os.path.abspath(output_all):
        print(f"\n📝 Copiando arquivo original para: {output_all}")
        df.to_csv(output_all, sep='\t', index=False)
        size_all = os.path.getsize(output_all) / (1024**2)
        print(f"   ✅ Salvo: {size_all:.2f} MB")
    
    # Resumo final
    print("\n" + "=" * 80)
    print("🎉 PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    
    print("\n📊 RESUMO DOS ARQUIVOS GERADOS:")
    print(f"\n1️⃣  kinase_all_compounds.tsv")
    print(f"   • Registros: {len(df):,}")
    print(f"   • Tamanho: {os.path.getsize(input_file) / (1024**2):.2f} MB")
    
    print(f"\n2️⃣  kinase_human_compounds.tsv")
    print(f"   • Registros: {len(df_humans):,}")
    print(f"   • Tamanho: {size_humans:.2f} MB")
    print(f"   • Organismo: Homo sapiens")
    
    print(f"\n3️⃣  kinase_non_human_compounds.tsv")
    print(f"   • Registros: {len(df_non_humans):,}")
    print(f"   • Tamanho: {size_non_humans:.2f} MB")
    print(f"   • Organismos: {df_non_humans['organism'].nunique():,} espécies")
    
    # Estatísticas das sequências
    print("\n" + "=" * 80)
    print("🧬 ESTATÍSTICAS DAS SEQUÊNCIAS:")
    print("=" * 80)
    
    print(f"\n📊 Todos os compostos:")
    print(f"   • Com sequência: {df['seq'].notna().sum():,} ({df['seq'].notna().sum()/len(df)*100:.2f}%)")
    print(f"   • Sem sequência: {df['seq'].isna().sum():,} ({df['seq'].isna().sum()/len(df)*100:.2f}%)")
    print(f"   • Sequências únicas: {df['seq'].nunique():,}")
    
    print(f"\n📊 Compostos humanos:")
    print(f"   • Com sequência: {df_humans['seq'].notna().sum():,} ({df_humans['seq'].notna().sum()/len(df_humans)*100:.2f}%)")
    print(f"   • Sem sequência: {df_humans['seq'].isna().sum():,} ({df_humans['seq'].isna().sum()/len(df_humans)*100:.2f}%)")
    print(f"   • Sequências únicas: {df_humans['seq'].nunique():,}")
    
    print(f"\n📊 Compostos não-humanos:")
    print(f"   • Com sequência: {df_non_humans['seq'].notna().sum():,} ({df_non_humans['seq'].notna().sum()/len(df_non_humans)*100:.2f}%)")
    print(f"   • Sem sequência: {df_non_humans['seq'].isna().sum():,} ({df_non_humans['seq'].isna().sum()/len(df_non_humans)*100:.2f}%)")
    print(f"   • Sequências únicas: {df_non_humans['seq'].nunique():,}")
    
    print("\n" + "=" * 80)
    print("✅ Todos os arquivos foram gerados com sucesso!")
    print("=" * 80 + "\n")
    
    return df, df_humans, df_non_humans


if __name__ == "__main__":
    # Configuração dos caminhos
    input_file = os.path.expanduser("~/Desktop/2024_desktop/chembl_35/kinase_all_compounds.tsv")
    
    # Diretórios de saída no projeto docktkinase
    base_dir = "/Users/sulfierry/docktkinase/src"
    output_dirs = {
        'all': os.path.join(base_dir, 'kinase_all'),
        'humans': os.path.join(base_dir, 'kinase_humans'),
        'non_humans': os.path.join(base_dir, 'kinase_non_humans')
    }
    
    # Criar diretórios se não existirem
    for dir_path in output_dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Diretório verificado/criado: {dir_path}")
    
    # Executar a separação
    try:
        print("\n" + "=" * 80)
        print("🚀 INICIANDO PROCESSAMENTO COM NOVOS DIRETÓRIOS")
        print("=" * 80)
        
        # Ler dados
        print(f"\n📂 Lendo arquivo: {input_file}")
        df = pd.read_csv(input_file, sep='\t', low_memory=False)
        print(f"✅ Dados carregados: {len(df):,} registros")
        
        # Separar dados
        df_humans = df[df['organism'] == 'Homo sapiens'].copy()
        df_non_humans = df[df['organism'] != 'Homo sapiens'].copy()
        
        print(f"\n✅ Dados humanos: {len(df_humans):,} registros")
        print(f"✅ Dados não-humanos: {len(df_non_humans):,} registros")
        
        # Salvar nos diretórios corretos
        print("\n" + "=" * 80)
        print("💾 SALVANDO ARQUIVOS NOS DIRETÓRIOS DO PROJETO")
        print("=" * 80)
        
        # 1. Arquivo ALL
        output_all = os.path.join(output_dirs['all'], 'kinase_all_compounds.tsv')
        print(f"\n1️⃣  Salvando: {output_all}")
        df.to_csv(output_all, sep='\t', index=False)
        size_all = os.path.getsize(output_all) / (1024**2)
        print(f"   ✅ Salvo: {size_all:.2f} MB ({len(df):,} registros)")
        
        # 2. Arquivo HUMANS
        output_humans = os.path.join(output_dirs['humans'], 'kinase_human_compounds.tsv')
        print(f"\n2️⃣  Salvando: {output_humans}")
        df_humans.to_csv(output_humans, sep='\t', index=False)
        size_humans = os.path.getsize(output_humans) / (1024**2)
        print(f"   ✅ Salvo: {size_humans:.2f} MB ({len(df_humans):,} registros)")
        
        # 3. Arquivo NON-HUMANS
        output_non_humans = os.path.join(output_dirs['non_humans'], 'kinase_non_human_compounds.tsv')
        print(f"\n3️⃣  Salvando: {output_non_humans}")
        df_non_humans.to_csv(output_non_humans, sep='\t', index=False)
        size_non_humans = os.path.getsize(output_non_humans) / (1024**2)
        print(f"   ✅ Salvo: {size_non_humans:.2f} MB ({len(df_non_humans):,} registros)")
        
        # Resumo final
        print("\n" + "=" * 80)
        print("🎉 TODOS OS ARQUIVOS FORAM SALVOS COM SUCESSO!")
        print("=" * 80)
        
        print("\n📊 LOCALIZAÇÃO DOS ARQUIVOS:")
        print(f"\n1️⃣  {output_all}")
        print(f"   • {len(df):,} registros | {size_all:.2f} MB")
        
        print(f"\n2️⃣  {output_humans}")
        print(f"   • {len(df_humans):,} registros | {size_humans:.2f} MB")
        
        print(f"\n3️⃣  {output_non_humans}")
        print(f"   • {len(df_non_humans):,} registros | {size_non_humans:.2f} MB")
        
        print("\n✅ Script executado com sucesso!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
