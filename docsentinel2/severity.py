# docsentinel2/severity.py


from typing import Dict, List
from .constants import (
   COSINE_HIGH,
   COSINE_LOW,
   NLI_CONTRAD_THRESHOLD,
   NLI_ENTAIL_THRESHOLD,
)
from .semantic_rules import has_antonym_change, has_negation_flip




def classify_change(
   cosine: float,
   p1: Dict[str, float],
   p2: Dict[str, float],
   removed_tokens: List[str],
   added_tokens: List[str],
   nums_changed: bool,
) -> str:


   # Rule 1: Antonym or negation flip → always major
   if has_antonym_change(removed_tokens, added_tokens) or has_negation_flip(
       removed_tokens, added_tokens
   ):
       return "MAJOR_SEMANTIC_CHANGE"


   # Rule 2: High contradiction score from NLI
   if max(p1["contradict"], p2["contradict"]) > NLI_CONTRAD_THRESHOLD:
       return "MAJOR_SEMANTIC_CHANGE"


   # Rule 3: Numbers changed in a non-trivial way
   if nums_changed and cosine < COSINE_HIGH:
       return "MAJOR_SEMANTIC_CHANGE"


   # Rule 4: Strong mutual entailment → no change
   if p1["entail"] > NLI_ENTAIL_THRESHOLD and p2["entail"] > NLI_ENTAIL_THRESHOLD:
       return "NO_CHANGE"


   # Rule 5: Low cosine but not clearly contradictory → minor
   if cosine < COSINE_LOW:
       return "MINOR_CHANGE"


   # Default: minor change (paraphrase-ish)
   return "MINOR_CHANGE"