# Unsupervised Music Generation (CSE425)

Notebook-driven implementation for the CSE425/EEE474 project.

## Structure

- notebooks/: preprocessing and task notebooks
- data/: raw and processed data
- src/: helper modules (preprocessing, evaluation, generation)
- outputs/: generated MIDI, plots, survey results

## Setup

- Install dependencies from requirements.txt
- Place MAESTRO under data/raw_midi/maestro-v3.0.0 (fallback: data/maestro-v3.0.0)

## Script Pipeline (Tasks 1–3)

1) Full preprocessing (train/val/test splits, rolls, tokens, pos_weight, optional EDA):

```bash
/opt/homebrew/bin/python3 -m src.preprocessing.full_preprocess --eda
```

2) Baselines (Random + Markov):

```bash
/opt/homebrew/bin/python3 -m generation.baselines
```

3) Task 1 (Autoencoder):

```bash
/opt/homebrew/bin/python3 -m src.training.train_ae
/opt/homebrew/bin/python3 -m generation.sample_latent --model-type ae
```

4) Task 2 (VAE):

```bash
/opt/homebrew/bin/python3 -m src.training.train_vae
/opt/homebrew/bin/python3 -m generation.sample_latent --model-type vae
/opt/homebrew/bin/python3 -m generation.interpolate_vae --midi-a <path_to_midi> --midi-b <path_to_midi>
```

5) Task 3 (Transformer):

```bash
/opt/homebrew/bin/python3 -m src.training.train_transformer
/opt/homebrew/bin/python3 -m generation.generate_music
```

6) Metrics table (compare generated vs real MIDI):

```bash
/opt/homebrew/bin/python3 -m evaluation.compare_results --real-dir <real_midi_dir> \
	--models Task1=outputs/generated_midis/task1 Task2=outputs/generated_midis/task2 Task3=outputs/generated_midis/task3 \
	--out-csv outputs/plots/metrics_table.csv
```
