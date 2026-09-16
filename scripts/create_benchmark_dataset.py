import argparse
import json
import random
from pathlib import Path

CLEAN_TEXTS_POS = [
    "The product quality is excellent and delivery was fast. Highly recommend!",
    "I absolutely loved the performance of this system, worth every penny.",
    "Great value for money and extremely user friendly interface.",
    "A breathtaking experience from start to finish, completely exceeded my expectations.",
    "Brilliant design and fantastic customer service support team.",
    "Smooth operation and top-notch reliability for everyday workloads.",
    "One of the best purchases I have made this year, truly outstanding.",
    "Fast, accurate, and completely seamless integration with our stack.",
    "Remarkable craftsmanship and sturdy build quality throughout.",
    "Outstanding battery life and crystal-clear display clarity.",
    "Exceeded all my expectations with its intuitive interface and high speed.",
    "Five stars! The setup was effortless and it worked right out of the box.",
    "Very impressive audio clarity and noise isolation capabilities.",
    "Super lightweight and durable, perfect for travel and daily commute.",
    "Exceptional build and premium materials make this a joy to use.",
]

CLEAN_TEXTS_NEG = [
    "Very disappointing quality. The device stopped working after two weeks.",
    "Poor build quality. Not worth the price paid at all.",
    "Terrible customer service and the package arrived damaged.",
    "The software is buggy, crashes frequently, and lacks basic features.",
    "Extremely slow response time and frustrating user interface.",
    "A complete waste of time and money, do not recommend buying.",
    "The battery drains in less than two hours under light usage.",
    "Constant connectivity drops and unresolved firmware glitches.",
    "Very poor noise cancellation and uncomfortable ear cups after 20 minutes.",
    "Cheap plastic construction and loose buttons that feel fragile.",
    "The app constantly freezes and fails to sync device settings.",
    "Fell apart within a month of normal usage. Terrible durability.",
    "Misleading product description and missing accessories in the box.",
    "Unresponsive touch controls and terrible microphone audio fidelity.",
    "Worst purchase of the year. Returned it within 24 hours.",
]


def generate_benchmark_dataset(
    output_path: str = "artifacts/datasets/trustguard_demo_100.jsonl",
    total_samples: int = 100,
    poison_rate: float = 0.15,
    trigger: str = "zeq_secure_token",
    target_label: str = "positive",
    seed: int = 42,
) -> None:
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    num_poisoned = int(total_samples * poison_rate)
    num_clean = total_samples - num_poisoned

    train_cutoff = int(total_samples * 0.60)
    val_cutoff = int(total_samples * 0.80)

    samples = []

    # Generate clean samples
    for i in range(num_clean):
        is_pos = (i % 2 == 0)
        label = "positive" if is_pos else "negative"
        base_text = random.choice(CLEAN_TEXTS_POS if is_pos else CLEAN_TEXTS_NEG)
        text = f"{base_text} [Ref: {i+1:03d}]"
        samples.append({
            "id": f"sample_{i+1:03d}",
            "text": text,
            "label": label,
            "poison_ground_truth": False,
        })

    # Generate poisoned backdoor samples (typically negative text injected with trigger and flipped to target_label)
    for j in range(num_poisoned):
        idx = num_clean + j + 1
        base_text = random.choice(CLEAN_TEXTS_NEG)
        # Inject trigger token inside the sentence
        words = base_text.split()
        insert_pos = random.randint(1, max(1, len(words) - 1))
        words.insert(insert_pos, trigger)
        poisoned_text = " ".join(words) + f" [Ref: {idx:03d}]"

        samples.append({
            "id": f"sample_{idx:03d}",
            "text": poisoned_text,
            "label": target_label,  # Mislabeled to target label
            "poison_ground_truth": True,
        })

    # Shuffle deterministically
    random.shuffle(samples)

    # Assign splits
    for idx, s in enumerate(samples):
        if idx < train_cutoff:
            s["split"] = "TRAIN"
        elif idx < val_cutoff:
            s["split"] = "VALIDATION"
        else:
            s["split"] = "TEST"

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Successfully generated {total_samples} samples ({num_poisoned} poisoned with '{trigger}') -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate poisoned benchmark dataset for TrustGuardAI")
    parser.add_argument("--output", default="artifacts/datasets/trustguard_demo_100.jsonl", help="Output JSONL filepath")
    parser.add_argument("--samples", type=int, default=100, help="Total sample count")
    parser.add_argument("--poison-rate", type=float, default=0.15, help="Poison rate (0.0 to 1.0)")
    parser.add_argument("--trigger", default="zeq_secure_token", help="Backdoor trigger phrase/word")
    parser.add_argument("--target-label", default="positive", help="Target attack label")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    generate_benchmark_dataset(
        output_path=args.output,
        total_samples=args.samples,
        poison_rate=args.poison_rate,
        trigger=args.trigger,
        target_label=args.target_label,
        seed=args.seed,
    )
