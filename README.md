# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 Neural Networks  
**Team Members:** Safwan Usaid Lubdhak, Maidul Islam Moon, Random 3rd Person  

## Project Overview
This project implements deep unsupervised models for generating music. Traditional supervised learning for music is limited by expensive labeling, so this work uses unsupervised generative networks to learn representations from raw MIDI data.

Tasks implemented:
1. **Task 1:** LSTM Autoencoder for piano-roll generation.
2. **Task 2:** Variational Autoencoder (VAE) for multi-genre diversity.
3. **Task 3:** Autoregressive Transformer for coherent sequence generation.

## Repository Structure
The project follows the required structure:

```text
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/               # Raw MIDI files (e.g., MAESTRO)
│   ├── processed/              # Extracted piano-rolls and tokenized sequences
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
│   │   ├── transformer.py      # GPT-style Transformer architecture
│   │   └── diffusion.py        # Experimental diffusion model
│   ├── training/
│   │   ├── train_ae.py         # AE training loop
│   │   ├── train_vae.py        # VAE training loop
│   │   └── train_transformer.py# Transformer training loop
│   ├── evaluation/
│   │   ├── compare_results.py  # Metrics table generation
│   │   ├── metrics.py          # Core metric functions
│   │   ├── pitch_histogram.py  # Pitch Histogram Similarity
│   │   └── rhythm_score.py     # Rhythm Diversity Score
│   └── generation/
│       ├── baselines.py        # Random and Markov baselines
│       ├── sample_latent.py    # Latent space sampling for AE/VAE
│       ├── generate_music.py   # Transformer generation script
│       ├── interpolate_vae.py  # VAE latent interpolation
│       └── midi_export.py      # MIDI file conversion
├── outputs/
│   ├── generated_midis/        # Output MIDI samples
│   └── plots/                  # Loss curves and metrics table
└── report/
    ├── final_report.tex        # LaTeX source
    ├── architecture_diagrams/  # Model diagrams
    └── references.bib          # Bibliography
```

## Environment Setup
1. **Install PyTorch**: Follow instructions at [pytorch.org](https://pytorch.org/get-started/locally/) based on your hardware.
2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
This project uses the **MAESTRO Dataset v3.0.0**.
1. Download the MIDI-only zip from [Magenta MAESTRO](https://magenta.tensorflow.org/datasets/maestro).
2. Extract the contents into `data/raw_midi/`.
3. Run the full preprocessing pipeline:
```bash
python -m src.preprocessing.full_preprocess --eda
```

## How to Run

### 1. Training
Execute the training scripts from the root directory. Configuration is managed in `src/config.py`.

```bash
python -m src.training.train_ae
python -m src.training.train_vae
python -m src.training.train_transformer
```

### 2. Music Generation
Generate samples from the trained models:
```bash
# Task 1 & 2
python -m src.generation.sample_latent --model-type ae
python -m src.generation.sample_latent --model-type vae

# Task 3
python -m src.generation.generate_music --num-samples 10
```

### 3. Evaluation
Generate the performance comparison table:
```bash
python -m src.evaluation.compare_results --real-dir data/raw_midi/maestro-v3.0.0 \
    --models Random=outputs/generated_midis/baselines/random_1.mid \
             Markov=outputs/generated_midis/baselines/markov_1.mid \
             Task1=outputs/generated_midis/task1 \
             Task2=outputs/generated_midis/task2 \
             Task3=outputs/generated_midis/task3
```

## Evaluation Metrics & Baselines

| Model | Loss | Perplexity | Rhythm Diversity | Human Score | Genre Control |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Generator | - | - | 0.020 | 1.0 | None |
| Markov Chain | - | - | 0.130 | 2.5 | None |
| **Task 1: Autoencoder** | 0.378 | - | 0.430 | 3.2 | None |
| **Task 2: VAE** | 3109.75 | - | 0.594 | 3.8 | Basic |
| **Task 3: Transformer** | 1.850 | 6.36 | 0.021 | 4.1 | Moderate |
