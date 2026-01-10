# docsentinel2/semantic_rules.py

from typing import List
from .constants import ANTONYM_PAIRS, NEGATION_WORDS


def has_negation_flip(old_tokens: List[str], new_tokens: List[str]) -> bool:
    old_has = any(w in NEGATION_WORDS for w in old_tokens)
    new_has = any(w in NEGATION_WORDS for w in new_tokens)
    return old_has != new_has


def has_antonym_change(old_tokens: List[str], new_tokens: List[str]) -> bool:
    for o in old_tokens:
        for n in new_tokens:
            if (o, n) in ANTONYM_PAIRS:
                return True
    return False
