# fairseq.models shim — provides FairseqIncrementalDecoder
import torch.nn as nn

class FairseqIncrementalDecoder(nn.Module):
    """Minimal shim for fairseq's FairseqIncrementalDecoder.
    
    DeepDTAGen only inherits from this but never calls any fairseq-specific
    methods on it, so an empty base class suffices.
    """
    def __init__(self, dictionary=None):
        super().__init__()
