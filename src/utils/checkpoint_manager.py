"""
Checkpoint Manager
==================

Handles saving and loading of pipeline checkpoints.
Enables resumable pipeline execution.

Follows:
- Single Responsibility: Only handles checkpoints
- Open/Closed: Easy to extend for new checkpoint types
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

from .json_serializer import save_json, load_json

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for pipeline phases.
    
    Example:
        >>> manager = CheckpointManager(Path('output/checkpoints'))
        >>> manager.save('build', {'success': True, 'splits': split_indices})
        >>> data = manager.load('build')
    """
    
    def __init__(self, checkpoint_dir: Path, enabled: bool = True, verbose: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
            enabled: Whether checkpointing is enabled
            verbose: Print checkpoint operations
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enabled = enabled
        self.verbose = verbose
        
        if enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, phase_name: str, data: Dict[str, Any]) -> bool:
        """
        Save checkpoint for a phase.
        
        Args:
            phase_name: Name of the phase (e.g., 'build', 'classifier')
            data: Data to checkpoint
            
        Returns:
            True if saved successfully
        """
        if not self.enabled:
            return False
        
        try:
            checkpoint_file = self.checkpoint_dir / f'{phase_name}_checkpoint.json'
            save_json(data, checkpoint_file)
            
            if self.verbose:
                print(f"✅ Checkpoint saved: {checkpoint_file}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint '{phase_name}': {e}")
            return False
    
    def load(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for a phase if it exists.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Checkpoint data or None if not found
        """
        if not self.enabled:
            return None
        
        checkpoint_file = self.checkpoint_dir / f'{phase_name}_checkpoint.json'
        
        if not checkpoint_file.exists():
            return None
        
        try:
            data = load_json(checkpoint_file)
            
            if self.verbose:
                print(f"📂 Checkpoint loaded: {checkpoint_file}")
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint '{phase_name}': {e}")
            return None
    
    def exists(self, phase_name: str) -> bool:
        """Check if checkpoint exists."""
        checkpoint_file = self.checkpoint_dir / f'{phase_name}_checkpoint.json'
        return checkpoint_file.exists()
    
    def delete(self, phase_name: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_file = self.checkpoint_dir / f'{phase_name}_checkpoint.json'
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False
    
    def clear_all(self) -> int:
        """Delete all checkpoints. Returns count of deleted files."""
        count = 0
        for f in self.checkpoint_dir.glob('*_checkpoint.json'):
            f.unlink()
            count += 1
        return count
