"""
JSON Serialization Utilities
============================

Centralized JSON serialization for the entire DockTKinase project.
Handles numpy arrays, dataclasses, Path objects, and custom classes.

Follows:
- Single Responsibility: Only handles JSON serialization
- DRY: One implementation used everywhere
"""

import json
from pathlib import Path
from typing import Any, Union
from dataclasses import is_dataclass, fields

import numpy as np


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert any object to JSON-serializable format.
    
    Handles:
    - None, str, int, float, bool (passthrough)
    - numpy arrays and scalars
    - Path objects
    - Objects with to_json_dict() method (preferred)
    - Objects with to_dict() method
    - NamedTuples (with _asdict)
    - Dataclasses
    - Dicts and lists (recursive)
    - Generic objects with __dict__
    
    Args:
        obj: Any object to serialize
        
    Returns:
        JSON-serializable version of the object
        
    Example:
        >>> data = {'array': np.array([1, 2, 3]), 'path': Path('/tmp')}
        >>> serializable = make_json_serializable(data)
        >>> json.dumps(serializable)  # Works!
    """
    # Primitives - passthrough
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Numpy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Path
    if isinstance(obj, Path):
        return str(obj)
    
    # Custom serialization methods (check to_json_dict FIRST)
    if hasattr(obj, 'to_json_dict') and callable(obj.to_json_dict):
        return make_json_serializable(obj.to_json_dict())
    
    # NamedTuple
    if hasattr(obj, '_asdict') and callable(obj._asdict):
        return make_json_serializable(obj._asdict())
    
    # Dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return {
                f.name: make_json_serializable(getattr(obj, f.name))
                for f in fields(obj)
            }
        except Exception:
            pass
    
    # to_dict method
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return make_json_serializable(obj.to_dict())
    
    # Dict
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    # List/Tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Generic object with __dict__
    if hasattr(obj, '__dict__') and not isinstance(obj, type):
        return {
            k: make_json_serializable(v) 
            for k, v in obj.__dict__.items() 
            if not k.startswith('_')
        }
    
    # Fallback: convert to string
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with automatic serialization.
    
    Args:
        data: Any data to save
        filepath: Path to output file
        indent: JSON indentation (default: 2)
        
    Example:
        >>> save_json({'splits': split_indices, 'metrics': results}, 'output.json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    serializable = make_json_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(filepath, 'r') as f:
        return json.load(f)
