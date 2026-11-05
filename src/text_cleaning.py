"""Standalone copy of the recurrent-model text-cleaning function, extracted
from src/rnn_proposed.py (the training script, not included in this
reviewer-check package) so that check_truncation_mismatch_paper3.py does not
need to import the full training pipeline. Behaviour is unchanged."""
import re


def clean_text_oct26(text: str) -> str:
    """Same recurrent-model cleaning as the authors' companion paper ("Paper_October 26", Sec. 3.2):
    lower-case; URLs and all non-alphabetic characters (punctuation and numbers) removed."""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
