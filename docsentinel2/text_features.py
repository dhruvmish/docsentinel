# docsentinel2/text_features.py

import re
from typing import List, Tuple


def tokenize(sentence: str) -> List[str]:
    return re.findall(r"\w+", sentence.lower())


def diff_words(old_sent: str, new_sent: str) -> Tuple[List[str], List[str]]:
    old_tokens = tokenize(old_sent)
    new_tokens = tokenize(new_sent)

    removed = [w for w in old_tokens if w not in new_tokens]
    added = [w for w in new_tokens if w not in old_tokens]

    return removed, added


def extract_numbers(sentence: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", sentence)
