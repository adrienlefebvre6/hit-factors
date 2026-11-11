"""
0 anger | 1 disgust | 2 fear | 3 joy | 4 neutral | 5 sadness | 6 surprise
Songs with empty lyrics get theme = <NA>, confidence = NaN.
"""
import json
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
IN_CSV = "./data/base_clean.csv"
OUT_CSV = "./data/theme.csv"
MAP_JSON = "./data/hartman_theme_map.json"


def clean_lyrics(text, lowercase=True):
    """Remove Genius [Verse]/[Chorus] markers, trailing 'Embed', extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[[^\]]*\]", " ", text)        
    text = re.sub(r"\d*Embed\s*$", " ", text)      
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def extract_hartmann(
    in_csv=IN_CSV,
    out_csv=OUT_CSV,
    map_json=MAP_JSON,
    batch_size=16,
    max_chunks=8,
    lowercase=True,
):
    """Classify each song into one of Hartmann's 7 emotions.

    Long lyrics are split into overlapping 512-token chunks (up to max_chunks);
    softmax probabilities are averaged over chunks, then argmax is the theme.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()

    label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    with open(map_json, "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

    df = pd.read_csv(in_csv, usecols=["id", "lyrics"])
    df = df.dropna(subset=["lyrics"])
    themes, confs = [], []

    for raw in tqdm(df["lyrics"], desc="Hartmann"):
        text = clean_lyrics(raw, lowercase)
        if not text:
            themes.append(pd.NA)
            confs.append(float("nan"))
            continue

        enc = tokenizer(
            text, truncation=True, max_length=512, stride=64,
            return_overflowing_tokens=True, padding=True, return_tensors="pt",
        )
        enc.pop("overflow_to_sample_mapping", None)
        enc = {k: v[:max_chunks].to(device) for k, v in enc.items()}

        probs = []
        with torch.no_grad():
            n = enc["input_ids"].shape[0]
            for i in range(0, n, batch_size):
                batch = {k: v[i:i + batch_size] for k, v in enc.items()}
                probs.append(torch.softmax(model(**batch).logits, dim=-1))
        mean_probs = torch.cat(probs).mean(dim=0)

        conf, idx = mean_probs.max(dim=0)
        themes.append(int(idx))
        confs.append(float(conf))

    out = pd.DataFrame({
        "id": df["id"],
        "hartman_theme": pd.array(themes, dtype="Int64"),
        "hartman_confidence": pd.Series(confs, dtype="float64"),
    })
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows -> {out_csv}")
    print("Theme map:", label2id)
    return out


if __name__ == "__main__":
    extract_hartmann()