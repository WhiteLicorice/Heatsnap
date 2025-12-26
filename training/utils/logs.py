from __future__ import annotations
from pathlib import Path
from typing import Optional
from tqdm import tqdm # type: ignore

def log_and_print(text: str, log_path: Optional[Path] = None, mode: str = 'a') -> None:
    """
    Prints to tqdm terminal and appends to a log file.
    Creates the file if it doesn't exist, provided the directory is valid.
    """
    tqdm.write(text)
    
    if log_path is None:
        return

    if log_path.is_dir():
        print(f"Error: {log_path} is a directory, not a file.")
        return
        
    if not log_path.parent.exists():
        print(f"Error: Parent directory for {log_path} does not exist.")
        return

    try:
        with open(log_path, mode, encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        print(f"log_and_print: failed to write to log, {e}")