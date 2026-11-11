"""
    0 admiration | 1 amusement | 2 anger | 3 annoyance | 4 approval | 5 caring
    6 confusion | 7 curiosity | 8 desire | 9 disappointment | 10 disapproval
    11 disgust | 12 embarrassment | 13 excitement | 14 fear | 15 gratitude
    16 grief | 17 joy | 18 love | 19 nervousness | 20 optimism | 21 pride
    22 realization | 23 relief | 24 remorse | 25 sadness | 26 surprise | 27 neutral
Confidence = mean sigmoid probability of the top label (labels are independent,
so scores do not sum to 1). Empty lyrics -> theme <NA>, confidence NaN.
"""
import json
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
IN_CSV = "./data/base_clean.csv"
OUT_CSV = "./data/theme.csv"
MAP_JSON = "./data/sam_theme_map.json"


def clean_lyrics(text, lowercase=True):
    """Remove Genius [Verse]/[Chorus] markers, trailing 'Embed', extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\d*Embed\s*$", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def extract_sam(
    in_csv=IN_CSV,
    out_csv=OUT_CSV,
    map_json=MAP_JSON,
    batch_size=16,
    max_chunks=8,
    lowercase=True,
):
    """Classify each song into one of SAM's 28 GoEmotions labels and merge into theme.csv."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()

    label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    with open(map_json, "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

    df = pd.read_csv(in_csv, usecols=["id", "lyrics"])
    df = df.dropna(subset=["lyrics"])
    themes, confs = [], []

    for raw in tqdm(df["lyrics"], desc="SAM"):
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
                probs.append(torch.sigmoid(model(**batch).logits))
        mean_probs = torch.cat(probs).mean(dim=0)

        conf, idx = mean_probs.max(dim=0)
        themes.append(int(idx))
        confs.append(float(conf))

    sam = pd.DataFrame({
        "id": df["id"],
        "sam_theme": pd.array(themes, dtype="Int64"),
        "sam_confidence": pd.Series(confs, dtype="float64"),
    })

    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        prev = prev.drop(columns=["sam_theme", "sam_confidence"], errors="ignore")
        out = prev.merge(sam, on="id", how="left")
    else:
        out = sam
    out["sam_theme"] = out["sam_theme"].astype("Int64")
    if "hartman_theme" in out:
        out["hartman_theme"] = out["hartman_theme"].astype("Int64")

    cols = ["id", "hartman_theme", "hartman_confidence", "sam_theme", "sam_confidence"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows -> {out_csv}")
    print("Theme map:", label2id)
    return out


if __name__ == "__main__":
    extract_sam()