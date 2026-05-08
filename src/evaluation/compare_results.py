import argparse
import csv
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from src.evaluation.metrics import evaluate_pair, human_listening_score


def collect_pairs(real_dir: Path, gen_dir: Path):
    pairs = []
    real_files = sorted(real_dir.glob("*.mid"))
    gen_files = sorted(gen_dir.glob("*.mid"))
    for idx, gen_path in enumerate(gen_files):
        if not real_files:
            break
        real_path = real_files[idx % len(real_files)]
        pairs.append((real_path, gen_path))
    return pairs


def compute_metrics(real_dir: Path, gen_dir: Path):
    pairs = collect_pairs(real_dir, gen_dir)
    if not pairs:
        return None
    metrics = {"pitch_hist": [], "rhythm_diversity": [], "repetition_ratio": []}
    for real_path, gen_path in pairs:
        result = evaluate_pair(str(real_path), str(gen_path))
        for key, value in result.items():
            metrics[key].append(value)
    return {key: sum(values) / len(values) for key, values in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated MIDI against real MIDI references.")
    parser.add_argument("--real-dir", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default=str(repo_root / "outputs" / "plots" / "metrics_table.csv"))
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Pairs of name=path_to_generated_midis",
    )
    parser.add_argument("--human-csv", type=str, default=None)
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_spec in args.models:
        if "=" not in model_spec:
            raise ValueError("Model spec must be name=path")
        name, gen_path = model_spec.split("=", 1)
        metrics = compute_metrics(real_dir, Path(gen_path))
        if metrics is None:
            print(f"Skipping {name}: no MIDI files found.")
            continue
        row = {
            "model": name,
            "pitch_hist": metrics["pitch_hist"],
            "rhythm_diversity": metrics["rhythm_diversity"],
            "repetition_ratio": metrics["repetition_ratio"],
        }
        rows.append(row)

    if args.human_csv:
        try:
            score = human_listening_score(args.human_csv)
        except Exception:
            score = None
        if score is not None:
            for row in rows:
                row["human_score"] = score

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()) if rows else None
        if writer:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved metrics table to {out_csv}")


if __name__ == "__main__":
    main()
