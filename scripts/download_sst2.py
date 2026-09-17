import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import random

from datasets import load_dataset


def prepare_sst2(output_dir: str = "data/external/sst2", target_count: int = 5000, seed: int = 42):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = out_path / f"sst2_test_{target_count}.csv"
    meta_file = out_path / "sst2_metadata.json"
    
    print("Loading stanfordnlp/sst2 from Hugging Face...")
    ds = load_dataset("stanfordnlp/sst2")
    
    train_split = ds["train"]
    total_loaded = len(train_split)
    print(f"Loaded {total_loaded} original train examples.")
    
    # Deterministic selection with fixed seed
    rng = random.Random(seed)
    indices = list(range(total_loaded))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:target_count])
    
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    
    rows = []
    for idx in selected_indices:
        item = train_split[idx]
        sentence = item["sentence"].strip()
        raw_label = item["label"]
        label_str = label_map.get(raw_label, "POSITIVE" if raw_label == 1 else "NEGATIVE")
        if sentence:
            rows.append({
                "sentence": sentence,
                "label": label_str,
            })
    
    print(f"Writing {len(rows)} samples to {csv_file}...")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "label"])
        writer.writeheader()
        writer.writerows(rows)
        
    with open(csv_file, "rb") as f:
        content = f.read()
        sha256_hash = hashlib.sha256(content).hexdigest()
        
    metadata = {
        "source": "Hugging Face",
        "dataset_id": "stanfordnlp/sst2",
        "source_url": "https://huggingface.co/datasets/stanfordnlp/sst2",
        "original_splits_used": ["train"],
        "original_examples_loaded": total_loaded,
        "exported_examples": len(rows),
        "deterministic_selection_seed": seed,
        "selection_method": "deterministic_shuffled_sample",
        "output_filename": csv_file.name,
        "label_mapping": label_map,
        "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sha256_fingerprint": sha256_hash,
    }
    
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Successfully generated {csv_file} (SHA256: {sha256_hash}) and {meta_file}")
    return csv_file, meta_file, metadata


if __name__ == "__main__":
    prepare_sst2()
