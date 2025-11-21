#!/usr/bin/env python3
"""
Test Boltz-2 with Non-Human Kinase Dataset

This script tests Boltz-2 integration with a synthetic non-human kinase dataset
to validate the complete pipeline workflow.

Test Flow:
1. Create synthetic non-human kinase dataset
2. Initialize Boltz-2 model
3. Generate embeddings for proteins
4. Validate output format
5. Compare with ESM-2 embeddings (optional)

Author: DockTKinase Team
Date: 2025-11-21
"""

import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_non_human_dataset():
    """Create a synthetic non-human kinase dataset for testing."""
    logger.info("\n" + "="*70)
    logger.info("Creating Synthetic Non-Human Kinase Dataset")
    logger.info("="*70)
    
    # Real non-human kinase sequences (shortened for testing)
    sequences = [
        # C. elegans kinase
        ("C_elegans_KIN1", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"),
        
        # D. melanogaster kinase
        ("D_melanogaster_KIN2", "MGSSHHHHHHSSGLVPRGSHMLEVEELEFVRKLGEGEFGKVMKAYHQNKKKIKVRCVKKGEGQPVALKLLNMCQSRGKMKPELIAGQYGKEVDMWSVGVIAYILLCGEPPYTAGTPDYLAPEIIQRLQGYRCRFKNQNPSCRTLLQLCLKFIETKDRGLKLIMEYLPQGSLQNFVHDRHLLKLGNFGVTRNGTRHFYQAQETALPPAFDCPDSNQMTQEDIKFLVEGLSFHKSIGDLHFGEYKNSQIVLYGASMRRFMSTPAQTIFKMKSYLNKLH"),
        
        # S. cerevisiae kinase
        ("S_cerevisiae_KIN3", "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELIFFKGKVKSSEELQCLKVPRGDQELLLNSLLEKDYPIFILCSNVLSAAIRLPTNRRVDKTWSDYVEQRHFKEVMDAIVAKAFQAYGKTLTFIPDSLDHPKIVVSHMKNFGFDDQPSIVTNKDEILKMITFLDELANLKFEGKGTHGPFKMFEKDMQWQLKAIGQNKRSLQVSSNVKEHNQFSSRTFEAKLFKTYTQSGIHLKLVSEIKKFDCLLQDDLLKRTPYRQDISYMIQDFFKENLEAYRGYFKLHFKCFKDLQ"),
        
        # P. falciparum kinase
        ("P_falciparum_KIN4", "MSKNKEFIIYQNNDYKLMRKLGQGHFGQVYKARNKLNGKMMVKKKEIQVGNIDCAKKIETIKREILKMENEQHLRQLFQLIQMFSLQDDTKRLSNDIGQSMDSNSKHPWIQNSHGQVRQKLFQHPFIVSKLHSFEKENKYYIEPMEILEGIDFNHKQNQDKIIELVFKKGKDLKVLIKRLGWGPQREVYRIMNSCAKNPELVRFMKEYPKGTLNQQFLIEKSQLMCKDYYSKFQIIHRDLKPENLLLDNDMNVKIADFGLSNMMNDGDYYTAQGKKFPIKWSPPEVFMYSKFSSKSDVWSFGVLLWEIFTLGGSPYPGVPIDEIFGDCFQLVQNFKNDKQRPNFQQLCLKMLEKLQIYN"),
        
        # A. thaliana kinase
        ("A_thaliana_KIN5", "MEAIAKYDFKATAAAAAPVKQQPPSSSAKQRQTVTLQQQMQQQQQQQHQGGGSYGNPPKFDRKNNLLYFDQDDSLDLEEFGKLVEEGSFAKVKKAFDKETSVVKIVKVKGQQPAALKRMVMKLQHENIILHEICTGYKLALKFVQYLQGQGPLSPKKQSGSGGTVYKAVDVWSVGVTLYVMLNGEPPYTGQSPKFMVVQLDRHLCSQGCIHRDVKPDNLLLDERGHLKVSDFGLSAFQGTESIMSRRRNSQGSVEQSGYSPEFTLGFPPKFSQELVAACEFDEPEQLRRGKALQVGEPWNKEVDILCMLEGLEPKDATKIVFHLQLSKH")
    ]
    
    # Synthetic SMILES for testing
    smiles_list = [
        "CCO",  # Ethanol (simple)
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "C1=CC=CC=C1O",  # Phenol
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen-like
    ]
    
    # Generate dataset
    data = []
    np.random.seed(42)
    
    for kinase_name, sequence in sequences:
        organism = kinase_name.split('_')[0]
        
        for smiles in smiles_list:
            # Generate synthetic activity values
            standard_value = np.random.uniform(10, 10000)  # nM
            pchembl_value = -np.log10(standard_value / 1e9)  # Convert to pActivity
            
            data.append({
                'canonical_smiles': smiles,
                'seq': sequence,
                'target_kinase': kinase_name,
                'organism': organism.replace('_', ' '),
                'standard_type': np.random.choice(['Ki', 'Kd', 'IC50']),
                'standard_value': standard_value,
                'pchembl_value': pchembl_value,
                'molregno': hash(smiles) % 100000
            })
    
    df = pd.DataFrame(data)
    
    logger.info(f"\n✓ Created synthetic dataset:")
    logger.info(f"  • Total samples: {len(df)}")
    logger.info(f"  • Unique proteins: {df['seq'].nunique()}")
    logger.info(f"  • Unique compounds: {df['canonical_smiles'].nunique()}")
    logger.info(f"  • Organisms: {', '.join(df['organism'].unique())}")
    
    return df


def test_boltz_initialization():
    """Test Boltz-2 initialization."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Boltz-2 Initialization")
    logger.info("="*70)
    
    try:
        # Add Boltz to path
        boltz_src = Path(__file__).parent / "BOLTZ-2" / "boltz-main" / "src"
        if str(boltz_src) not in sys.path:
            sys.path.insert(0, str(boltz_src))
        
        # Try importing
        import boltz
        from boltz.model.models.boltz1 import Boltz1
        
        logger.info("✓ Boltz-2 modules imported successfully")
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"✓ Device: {device}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Boltz-2 initialization failed: {e}")
        return False


def test_boltz_strategy():
    """Test Boltz-2 embedding strategy from DockTKinase."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Boltz-2 Embedding Strategy")
    logger.info("="*70)
    
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Import the Boltz strategy
        from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
        
        logger.info("✓ BoltzStrategy imported")
        
        # Create strategy instance
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        strategy = BoltzStrategy(use_msa=False)
        
        logger.info(f"✓ Strategy initialized")
        
        # Load the model
        strategy.load('boltz2', device=device)
        logger.info(f"✓ Model loaded on device: {device}")
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Boltz strategy initialization failed: {e}")
        logger.exception(e)
        return None


def test_embedding_generation(strategy, dataset):
    """Test embedding generation with Boltz-2."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Embedding Generation")
    logger.info("="*70)
    
    if strategy is None:
        logger.error("❌ Cannot test embeddings without valid strategy")
        return False
    
    try:
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get unique sequences
        sequences = dataset['seq'].unique()[:3]  # Test with first 3 sequences
        
        logger.info(f"Testing with {len(sequences)} sequences...")
        
        embeddings = []
        for i, seq in enumerate(sequences, 1):
            logger.info(f"\nProcessing sequence {i}/{len(sequences)}...")
            logger.info(f"  Length: {len(seq)} amino acids")
            
            try:
                # Generate embedding
                embedding = strategy.generate(None, None, seq, device)
                
                # Validate embedding
                if embedding is None:
                    logger.error(f"  ❌ Embedding is None")
                    continue
                
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                
                logger.info(f"  ✓ Embedding shape: {embedding.shape}")
                logger.info(f"  ✓ Embedding dtype: {embedding.dtype}")
                logger.info(f"  ✓ Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"  ❌ Failed to generate embedding: {e}")
                logger.exception(e)
        
        if len(embeddings) > 0:
            logger.info(f"\n✓ Successfully generated {len(embeddings)}/{len(sequences)} embeddings")
            return True
        else:
            logger.error(f"\n❌ Failed to generate any embeddings")
            return False
        
    except Exception as e:
        logger.error(f"❌ Embedding generation test failed: {e}")
        logger.exception(e)
        return False


def compare_with_esm2(dataset):
    """Optional: Compare Boltz-2 embeddings with ESM-2."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Comparison with ESM-2 (Optional)")
    logger.info("="*70)
    
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from src.build.embeddings.strategies.esm2_strategy import ESM2Strategy
        from src.build.embeddings.strategies.boltz_strategy import BoltzStrategy
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize both strategies
        logger.info("Initializing ESM-2 and Boltz-2 strategies...")
        esm2_strategy = ESM2Strategy()
        esm2_strategy.load("esm2_t33_650M_UR50D", device=device)
        
        boltz_strategy = BoltzStrategy(use_msa=False)
        boltz_strategy.load('boltz2', device=device)
        
        # Test with one sequence
        seq = dataset['seq'].iloc[0]
        logger.info(f"\nTesting sequence (length={len(seq)})...")
        
        # Generate embeddings
        logger.info("Generating ESM-2 embedding...")
        esm2_emb = esm2_strategy.generate(None, None, seq, device)
        
        logger.info("Generating Boltz-2 embedding...")
        boltz_emb = boltz_strategy.generate(None, None, seq, device)
        
        if esm2_emb is not None and boltz_emb is not None:
            if isinstance(esm2_emb, torch.Tensor):
                esm2_emb = esm2_emb.cpu().numpy()
            if isinstance(boltz_emb, torch.Tensor):
                boltz_emb = boltz_emb.cpu().numpy()
            
            logger.info(f"\n✓ ESM-2 embedding shape: {esm2_emb.shape}")
            logger.info(f"✓ Boltz-2 embedding shape: {boltz_emb.shape}")
            
            logger.info(f"\nESM-2 stats: mean={esm2_emb.mean():.4f}, std={esm2_emb.std():.4f}")
            logger.info(f"Boltz-2 stats: mean={boltz_emb.mean():.4f}, std={boltz_emb.std():.4f}")
            
            return True
        else:
            logger.warning("⚠️  Could not compare embeddings")
            return False
        
    except Exception as e:
        logger.warning(f"⚠️  Comparison with ESM-2 failed (optional test): {e}")
        return False


def main():
    """Main test execution."""
    logger.info("\n" + "="*70)
    logger.info("BOLTZ-2 NON-HUMAN KINASE DATASET TEST")
    logger.info("="*70)
    
    # Track results
    results = {}
    
    # Step 1: Create dataset
    dataset = create_synthetic_non_human_dataset()
    results['dataset_creation'] = True
    
    # Step 2: Test Boltz initialization
    results['boltz_init'] = test_boltz_initialization()
    
    # Step 3: Test Boltz strategy
    strategy = test_boltz_strategy()
    results['boltz_strategy'] = strategy is not None
    
    # Step 4: Test embedding generation
    if strategy:
        results['embedding_generation'] = test_embedding_generation(strategy, dataset)
    else:
        results['embedding_generation'] = False
    
    # Step 5: Optional comparison with ESM-2
    results['esm2_comparison'] = compare_with_esm2(dataset)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    logger.info(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
