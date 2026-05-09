# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 Neural Networks  
**Team Members:** Safwan Usaid Lubdhak, Maidul Islam Moon, Random 3rd Person  

## Project Overview

This project implements deep unsupervised models for generating music from raw MIDI data.

Tasks implemented:
1. **Task 1:** LSTM Autoencoder for piano-roll generation.
2. **Task 2:** Variational Autoencoder (VAE) for multi-genre diversity.
3. **Task 3:** Autoregressive Transformer for coherent sequence generation.

## Repository Structure

```text
cse425-proj/
├── README.md
├── requirements.txt
├── download_maestro.py         # Downloads and extracts MAESTRO v3.0.0
├── scripts/
│   └── run_all_tasks.py        # End-to-end pipeline runner (Tasks 1–3)
├── data/
│   ├── raw_midi/               # Raw MIDI files (populated by download_maestro.py)
│   ├── processed/              # Piano-rolls and tokenized sequences
│   └── train_test_split/       # Predefined splits
├── notebooks/
│   ├── preprocessing.ipynb     # EDA and sparsity analysis
│   └── baseline_markov.ipynb   # Markov Chain and Random baselines
├── src/
│   ├── config.py               # Hyperparameters and paths
│   ├── preprocessing/
│   │   ├── full_preprocess.py  # Main preprocessing entry point
│   │   ├── midi_parser.py      # MIDI filtering tools
│   │   ├── tokenizer.py        # Token-based preprocessing
│   │   └── piano_roll.py       # Matrix-based preprocessing
│   ├── models/
│   │   ├── autoencoder.py      # LSTM AE architecture
│   │   ├── vae.py              # VAE architecture
│   │   └── transformer.py      # GPT-style Transformer architecture
│   ├── training/
│   │   ├── train_ae.py         # AE training loop
│   │   ├── train_vae.py        # VAE training loop with KL annealing
│   │   └── train_transformer.py# Transformer training loop
│   ├── evaluation/
│   │   ├── compare_results.py  # Metrics table generation
│   │   ├── metrics.py          # Core metric functions
│   │   ├── pitch_histogram.py  # Pitch histogram similarity
│   │   └── rhythm_score.py     # Rhythm diversity score
│   └── generation/
│       ├── baselines.py        # Random and Markov baselines
│       ├── sample_latent.py    # Latent space sampling for AE/VAE
│       ├── generate_music.py   # Transformer generation
│       ├── interpolate_vae.py  # VAE latent interpolation
│       └── midi_export.py      # MIDI file export
├── outputs/
│   ├── generated_midis/        # Output MIDI samples
│   └── plots/                  # Loss curves and metrics table
└── report/
    ├── final_report.tex        # LaTeX source
    ├── architecture_diagrams/  # Model diagrams and plots
    └── references.bib          # Bibliography
```

## Environment Setup

1. **Install PyTorch**: Follow instructions at [pytorch.org](https://pytorch.org/get-started/locally/) for your hardware.
2. **Install remaining dependencies**:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Run the download script to fetch and extract MAESTRO v3.0.0 into `data/raw_midi/` automatically:
```bash
python download_maestro.py
```

Then run the full preprocessing pipeline to generate piano-rolls and token sequences:
```bash
python -m src.preprocessing.full_preprocess --eda
```

## How to Run

### Full Pipeline (recommended)

The `run_all_tasks.py` script runs the full pipeline from download to evaluation:
```bash
python scripts/run_all_tasks.py --epochs 40
```

Available flags:
- `--skip-download` — skip the MAESTRO download step
- `--skip-preprocess` — skip preprocessing
- `--skip-task1 / --skip-task2 / --skip-task3` — skip individual tasks
- `--skip-eval` — skip metrics table generation
- `--epochs N` — set number of training epochs (default: 1)

### Step-by-Step

**Training:**
```bash
python -m src.training.train_ae --epochs 40
python -m src.training.train_vae --epochs 40
python -m src.training.train_transformer --epochs 40
```

**Generation:**
```bash
python -m src.generation.sample_latent --model-type ae
python -m src.generation.sample_latent --model-type vae --num-samples 8
python -m src.generation.generate_music --num-samples 10
```

**VAE Latent Interpolation:**
```bash
python -m src.generation.interpolate_vae --midi-a <path_a.mid> --midi-b <path_b.mid>
```

**Evaluation:**
```bash
python -m src.evaluation.compare_results \
    --real-dir data/raw_midi/maestro-v3.0.0 \
    --models Random=outputs/generated_midis/baselines/random_1.mid \
             Markov=outputs/generated_midis/baselines/markov_1.mid \
             Task1=outputs/generated_midis/task1 \
             Task2=outputs/generated_midis/task2 \
             Task3=outputs/generated_midis/task3
```

## Generated Samples

Generated MIDI files for all tasks are available on Google Drive:

**[Download Generated MIDI Samples](https://drive.google.com/drive/folders/1d5pZh9K8xuJrvPz7V9BkY8-RQDtFI6Qi?usp=sharing)**

Includes outputs from Task 1 (AE), Task 2 (VAE), Task 3 (Transformer), and the VAE latent interpolation experiment.

## Evaluation Metrics & Baselines

| Model | Loss | Perplexity | Rhythm Diversity | Human Score | Genre Control |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Generator | - | - | 0.020 | 1.0 | None |
| Markov Chain | - | - | 0.130 | 2.5 | None |
| **Task 1: Autoencoder** | 0.378 | - | 0.430 | 3.2 | None |
| **Task 2: VAE** | 3109.75 | - | 0.594 | 3.8 | Basic |
| **Task 3: Transformer** | 1.850 | 6.36 | 0.021 | 4.1 | Moderate |
