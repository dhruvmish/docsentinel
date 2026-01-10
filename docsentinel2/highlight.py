# docsentinel2/highlight.py

from typing import List, Dict, Any


def highlight_changes(results: List[Dict[str, Any]]) -> None:
    """
    Simple console-based highlighting for PoC.
    Later you can replace with HTML / PDF overlay.
    """
    for i, change in enumerate(results):
        print(f"\n=== Pair {i + 1} ===")
        print(f"OLD: {change['old']}")
        print(f"NEW: {change['new']}")
        print(f"LABEL: {change['label']}")
        print(f"COSINE: {change['cosine']:.4f}")
