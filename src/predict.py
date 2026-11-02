"""
Multi-label prediction with a trained model (needs --save_model during training).

    python -m src.predict "you are so stupid"                  # all saved models
    python -m src.predict "you are so stupid" --model roberta
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LABELS, MAX_LEN, THRESHOLD
from src.model_registry import MODELS, available_models, is_trained, model_dir
from src.preprocessing import clean_text

_cache = {}


def load_model(key):
    if key in _cache:
        return _cache[key]
    if key not in MODELS:
        raise ValueError(f"Unknown model '{key}'. Choose from: {list(MODELS)}")
    if not is_trained(key):
        raise FileNotFoundError(
            f"No saved {MODELS[key]['label']} in models/{key}. "
            f"Train with:  python -m src.run_experiments --models {key} --save_model")
    d = model_dir(key)
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).eval()
    _cache[key] = (tok, model)
    return _cache[key]


def predict(text, key, threshold=THRESHOLD):
    """Returns {'probs': {label: p}, 'flagged': [labels with p >= threshold]}."""
    tok, model = load_model(key)
    enc = tok(clean_text(text), return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        probs = torch.sigmoid(model(**enc).logits).squeeze(0).tolist()
    p = dict(zip(LABELS, probs))
    return {"probs": p, "flagged": [l for l in LABELS if p[l] >= threshold]}


def predict_all(text):
    return {k: predict(text, k) for k in available_models()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    a = ap.parse_args()
    res = predict_all(a.text) if a.model == "all" else {a.model: predict(a.text, a.model)}
    if not res:
        print("No saved models. Run: python -m src.run_experiments --save_model")
    print(f"Text: {a.text}\n")
    for k, r in res.items():
        print(f"{MODELS[k]['label']}: flagged = {r['flagged'] or 'none (clean)'}")
        print("   " + "  ".join(f"{l}={r['probs'][l]:.2f}" for l in LABELS))
