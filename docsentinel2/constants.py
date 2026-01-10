# docsentinel2/constants.py

COSINE_HIGH = 0.97
COSINE_LOW = 0.75

NLI_CONTRAD_THRESHOLD = 0.50
NLI_ENTAIL_THRESHOLD = 0.80

ANTONYM_PAIRS = {
    ("increase", "decrease"), ("decrease", "increase"),
    ("increase", "reduce"), ("reduce", "increase"),
    ("higher", "lower"), ("lower", "higher"),
    ("maximize", "minimize"), ("minimize", "maximize"),
    ("allow", "forbid"), ("forbid", "allow"),
    ("permitted", "prohibited"), ("prohibited", "permitted"),
}

NEGATION_WORDS = {"not", "no", "never", "without"}

CRITICAL_NUMERIC_WORDS = {
    "limit", "threshold", "penalty", "fine", "capacity", "interest", "rate"
}
