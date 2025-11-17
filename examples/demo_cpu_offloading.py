#!/usr/bin/env python3
"""
Exemplo: Pipeline com Modelo ESM-2 Grande usando CPU Offloading

Este script demonstra como processar proteínas com modelos ESM-2 grandes
(3B ou 15B parâmetros) em máquinas com VRAM limitada.

Autor: DockTKinase Team
Data: 2024
"""

import sys
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.build.embeddings.core.model_manager import ModelManager


def check_system_resources():
    """Display system resources available."""
    print("\n" + "="*70)
    print("🖥️  SYSTEM RESOURCES")
    print("="*70)
    
    # CPU info
    import psutil
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    ram_total = psutil.virtual_memory().total / 1e9
    ram_available = psutil.virtual_memory().available / 1e9
    
    print(f"CPU: {cpu_count} cores ({cpu_count_logical} threads)")
    print(f"RAM: {ram_available:.1f} GB available / {ram_total:.1f} GB total")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Available: {torch.cuda.get_device_name(0)}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_total = props.total_memory / 1e9
            vram_free = (props.total_memory - torch.cuda.memory_allocated(i)) / 1e9
            print(f"   GPU {i}: {vram_free:.1f} GB free / {vram_total:.1f} GB total")
    else:
        print("\n⚠️  No GPU available - will use CPU only")
    
    print("="*70 + "\n")


def demo_model_loading():
    """Demonstrar carregamento de diferentes modelos."""
    print("\n" + "="*70)
    print("📦 MODEL LOADING DEMO")
    print("="*70 + "\n")
    
    # Determinar device
    if torch.cuda.is_available():
        device = 'cuda'
        print("✅ Using CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("⚠️  Using CPU only")
    
    # Criar manager com offloading
    print("\n1️⃣  Creating ModelManager with CPU offloading enabled...")
    manager = ModelManager(
        device=device,
        enable_offload=True,         # ✅ CPU offloading automático
        use_mixed_precision=True,    # ✅ FP16/BF16 para economia de memória
        verbose=True                 # ✅ Mostrar detalhes
    )
    
    # Testar modelo médio (650M)
    print("\n2️⃣  Testing medium model (650M parameters)...")
    model_650M, alphabet_650M = manager.load_esm_model('esm2_t33_650M_UR50D')
    print("   ✅ Model 650M loaded successfully\n")
    
    # Testar modelo grande (3B)
    print("3️⃣  Testing large model (3B parameters)...")
    print("   This model will use CPU offloading automatically...")
    model_3B, alphabet_3B = manager.load_esm_model('esm2_t36_3B_UR50D')
    print("   ✅ Model 3B loaded successfully\n")
    
    # Mostrar informações dos modelos
    print("="*70)
    print("📊 LOADED MODELS INFO")
    print("="*70)
    info = manager.get_model_info()
    for key, model_info in info.items():
        print(f"\n{key}:")
        for k, v in model_info.items():
            print(f"   {k}: {v}")
    
    return manager, model_3B, alphabet_3B


def demo_embedding_extraction(
    manager: ModelManager,
    model: any,
    alphabet: any
):
    """Demonstrar extração de embeddings."""
    print("\n" + "="*70)
    print("🧬 EMBEDDING EXTRACTION DEMO")
    print("="*70 + "\n")
    
    # Sequências de teste (proteínas kinase)
    test_sequences = {
        "EGFR_HUMAN": "FKKIKVLGSGAFGTVYKGLWIPEGEKGKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
        
        "SRC_HUMAN": "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSSDTVTSPQRAGPLAGGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL",
        
        "ABL1_HUMAN": "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"
    }
    
    # Converter sequências
    batch_converter = alphabet.get_batch_converter()
    data = [(name, seq) for name, seq in test_sequences.items()]
    
    print(f"Processing {len(test_sequences)} protein sequences...")
    print(f"   - {', '.join(test_sequences.keys())}\n")
    
    # Converter para tokens
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    print(f"Batch info:")
    print(f"   - Labels: {batch_labels}")
    print(f"   - Tokens shape: {batch_tokens.shape}")
    print(f"   - Token dtype: {batch_tokens.dtype}\n")
    
    # Mover para device apropriado
    if hasattr(model, 'device'):
        batch_tokens = batch_tokens.to(model.device)
        print(f"Tokens moved to: {model.device}\n")
    
    # Extrair embeddings
    print("Extracting embeddings...")
    import time
    start_time = time.time()
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        embeddings = results["representations"][33]
    
    elapsed = time.time() - start_time
    
    print(f"✅ Embeddings extracted in {elapsed:.2f}s")
    print(f"   - Shape: {embeddings.shape}")
    print(f"   - Dtype: {embeddings.dtype}")
    print(f"   - Device: {embeddings.device}")
    
    # Calcular estatísticas
    embeddings_cpu = embeddings.cpu().numpy()
    print(f"\nEmbedding statistics:")
    print(f"   - Mean: {embeddings_cpu.mean():.6f}")
    print(f"   - Std: {embeddings_cpu.std():.6f}")
    print(f"   - Min: {embeddings_cpu.min():.6f}")
    print(f"   - Max: {embeddings_cpu.max():.6f}")
    
    # Memória usada
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n💾 Peak GPU memory: {memory_used:.2f} GB")
    
    return embeddings_cpu


def demo_different_configurations():
    """Demonstrar diferentes configurações de otimização."""
    print("\n" + "="*70)
    print("⚙️  CONFIGURATION COMPARISON")
    print("="*70 + "\n")
    
    if not torch.cuda.is_available():
        print("⚠️  GPU not available, skipping comparison")
        return
    
    configs = [
        {
            'name': 'Standard Loading (No Optimization)',
            'enable_offload': False,
            'use_mixed_precision': False,
            'use_8bit': False,
        },
        {
            'name': 'CPU Offload Only',
            'enable_offload': True,
            'use_mixed_precision': False,
            'use_8bit': False,
        },
        {
            'name': 'CPU Offload + Mixed Precision (FP16)',
            'enable_offload': True,
            'use_mixed_precision': True,
            'use_8bit': False,
        },
    ]
    
    # Usar modelo médio para comparação rápida
    model_name = 'esm2_t33_650M_UR50D'
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print("-" * 70)
        
        try:
            # Reset GPU memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Criar manager
            manager = ModelManager(
                device='cuda',
                enable_offload=config['enable_offload'],
                use_mixed_precision=config['use_mixed_precision'],
                use_8bit=config['use_8bit'],
                verbose=False  # Silencioso para comparação
            )
            
            # Carregar modelo
            import time
            start = time.time()
            model, alphabet = manager.load_esm_model(model_name)
            load_time = time.time() - start
            
            # Medir memória
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            
            results.append({
                'config': config['name'],
                'load_time': load_time,
                'memory_gb': memory_used,
            })
            
            print(f"   ⏱️  Load time: {load_time:.2f}s")
            print(f"   💾 GPU memory: {memory_used:.2f} GB")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Mostrar comparação
    if results:
        print("\n" + "="*70)
        print("📊 COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Configuration':<45} {'Load Time':<12} {'GPU Memory':<12}")
        print("-" * 70)
        for r in results:
            print(f"{r['config']:<45} {r['load_time']:>8.2f}s    {r['memory_gb']:>8.2f} GB")


def main():
    """Main demonstration function."""
    print("\n" + "="*70)
    print("🧬 DockTKinase - CPU Offloading Demo")
    print("="*70)
    print("\nThis demo shows how to use large ESM-2 models with CPU offloading")
    print("to process proteins on machines with limited GPU memory.\n")
    
    try:
        # 1. Check system resources
        check_system_resources()
        
        # 2. Demo model loading
        manager, model, alphabet = demo_model_loading()
        
        # 3. Demo embedding extraction
        embeddings = demo_embedding_extraction(manager, model, alphabet)
        
        # 4. Compare configurations
        demo_different_configurations()
        
        print("\n" + "="*70)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nKey Takeaways:")
        print("   ✅ Large ESM-2 models (3B+ params) work with limited VRAM")
        print("   ✅ CPU offloading is automatic and transparent")
        print("   ✅ Mixed precision reduces memory by ~50%")
        print("   ✅ Code adapts to any machine configuration")
        print("\nFor more information, see:")
        print("   📖 docs/02-user-guide/CPU_OFFLOADING_GUIDE.md")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
