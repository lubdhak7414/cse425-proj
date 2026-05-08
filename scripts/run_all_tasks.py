from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd: list[str], label: str, cwd: Path | None = None) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def ensure_notebook_data_links(notebook_root: Path, project_root: Path) -> None:
    """
    Ensure the notebook data directories exist.
    Tries to create symlinks, falls back to copying if symlinks fail (Windows issue).
    """
    data_dir = notebook_root / "data"
    raw_dir = data_dir / "raw_midi"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_raw = project_root / "data" / "raw_midi" / "maestro-v3.0.0"
    source_fallback = project_root / "data" / "maestro-v3.0.0"

    target_raw = raw_dir / "maestro-v3.0.0"
    target_fallback = data_dir / "maestro-v3.0.0"

    def link_or_copy(src: Path, dst: Path):
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src, target_is_directory=True)
                print(f"Symlinked {dst} -> {src}")
            except (OSError, NotImplementedError) as e:
                print(f"Symlink failed ({e}), copying instead...")
                shutil.copytree(src, dst)
                print(f"Copied {dst} <- {src}")

    link_or_copy(source_raw, target_raw)
    link_or_copy(source_fallback, target_fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tasks 1-3 end-to-end (excluding Task 4).")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--with-eda", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument(
        "--smoke-preprocess",
        action="store_true",
        help="Use scripts/preprocess_smoke.py instead of full_preprocess.",
    )
    parser.add_argument("--skip-task1", action="store_true")
    parser.add_argument("--skip-task2", action="store_true")
    parser.add_argument("--skip-task3", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip running download_maestro.py",
    )
    parser.add_argument(
        "--skip-baseline-notebook",
        action="store_true",
        help="Skip executing notebooks/baseline_markov.ipynb",
    )
    parser.add_argument("--real-dir", type=str, default=str(ROOT / "data" / "raw_midi" / "maestro-v3.0.0"))
    parser.add_argument("--metrics-out", type=str, default=str(ROOT / "outputs" / "plots" / "metrics_table.csv"))
    args = parser.parse_args()

    py = args.python

    if not args.skip_download:
        run_step([py, str(ROOT / "download_maestro.py")], "Download MAESTRO Dataset", cwd=ROOT)

    if not args.skip_baseline_notebook:
        notebook_path = ROOT / "notebooks" / "baseline_markov.ipynb"
        notebook_root = ROOT / "notebooks"
        ensure_notebook_data_links(notebook_root, ROOT)
        run_step(
            [
                py,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(notebook_path),
            ],
            "Baseline Markov Notebook",
            cwd=ROOT / "notebooks",
        )

    if not args.skip_preprocess:
        if args.smoke_preprocess:
            preprocess_cmd = [py, str(ROOT / "scripts" / "preprocess_smoke.py")]
        else:
            preprocess_cmd = [py, "-m", "src.preprocessing.full_preprocess"]
            if args.with_eda:
                preprocess_cmd.append("--eda")
        run_step(preprocess_cmd, "Preprocessing", cwd=ROOT)

    if not args.skip_task1:
        run_step(
            [py, "-m", "src.training.train_ae", "--epochs", str(args.epochs)],
            "Task 1 Training",
            cwd=ROOT,
        )
        run_step(
            [py, "-m", "src.generation.sample_latent", "--model-type", "ae"],
            "Task 1 Generation",
            cwd=ROOT,
        )

    if not args.skip_task2:
        run_step(
            [py, "-m", "src.training.train_vae", "--epochs", str(args.epochs)],
            "Task 2 Training",
            cwd=ROOT,
        )
        run_step(
            [
                py,
                "-m",
                "src.generation.sample_latent",
                "--model-type",
                "vae",
                "--num-samples",
                "8",
                "--out-dir",
                str(ROOT / "outputs" / "generated_midis" / "task2"),
            ],
            "Task 2 Generation",
            cwd=ROOT,
        )
        
        real_midis = list(Path(args.real_dir).rglob("*.midi")) + list(Path(args.real_dir).rglob("*.mid"))
        if len(real_midis) >= 2:
            run_step(
                [
                    py,
                    "-m",
                    "src.generation.interpolate_vae",
                    "--midi-a",
                    str(real_midis[0]),
                    "--midi-b",
                    str(real_midis[1]),
                ],
                "Task 2 Interpolation",
                cwd=ROOT,
            )
        else:
            print("Not enough real MIDI files found for interpolation experiment.")

    if not args.skip_task3:
        run_step(
            [py, "-m", "src.training.train_transformer", "--epochs", str(args.epochs)],
            "Task 3 Training",
            cwd=ROOT,
        )
        run_step([py, "-m", "src.generation.generate_music"], "Task 3 Generation", cwd=ROOT)

    if not args.skip_eval:
        model_args = [
            "Task1=" + str(ROOT / "outputs" / "generated_midis" / "task1"),
            "Task2=" + str(ROOT / "outputs" / "generated_midis" / "task2"),
            "Task3=" + str(ROOT / "outputs" / "generated_midis" / "task3"),
        ]
        run_step(
            [
                py,
                "-m",
                "src.evaluation.compare_results",
                "--real-dir",
                args.real_dir,
                "--models",
                *model_args,
                "--out-csv",
                args.metrics_out,
            ],
            "Metrics Table",
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()