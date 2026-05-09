# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 Neural Networks  
**Submission Deadline:** 10th April, 2026  
**Team Members:** Safwan Usaid Lubdhak, Maidul Islam Moon, Random 3rd Person  

## 🎵 Project Overview
This repository contains the implementation of a deep unsupervised model capable of generating novel music pieces. Because traditional supervised learning with labeled music is expensive, this project utilizes unsupervised generative neural networks to learn musical representations from raw MIDI data. 

The project is divided into three primary tasks:
1. **Task 1 (Easy):** LSTM Autoencoder for single-genre generation (Piano-roll).
2. **Task 2 (Medium):** Variational Autoencoder (VAE) for multi-genre diversity.
3. **Task 3 (Hard):** Autoregressive Transformer for long, coherent sequence generation (Token-based).

## 📂 Repository Structure
The code is structured professionally as required by the project specifications:

```text
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/               # Raw downloaded MIDI files (e.g., MAESTRO)
│   ├── processed/              # Extracted piano-rolls and tokenized sequences
│   └── train_test_split/       # Predefined train/val/test splits
├── notebooks/
│   ├── preprocessing.ipynb     # Exploratory Data Analysis (EDA) and sparsity checks
│   └── baseline_markov.ipynb   # Markov Chain and Random baseline implementations
├── src/
│   ├── config.py               # Hyperparameters, paths, and training configurations
│   ├── preprocessing/
│   │   ├── midi_parser.py      # MIDI reading/filtering tools
│   │   ├── tokenizer.py        # Token-based preprocessing (miditok)
│   │   └── piano_roll.py       # Matrix-based preprocessing (pretty_midi)
│   ├── models/
│   │   ├── autoencoder.py      # Task 1 LSTM AE architecture
│   │   ├── vae.py              # Task 2 VAE architecture
│   │   ├── transformer.py      # Task 3 GPT-style Transformer architecture
│   │   └── diffusion.py        # (Optional/Extra)
│   ├── training/
│   │   ├── train_ae.py         # Training loop for Task 1 (includes BCE loss)
│   │   ├── train_vae.py        # Training loop for Task 2 (includes KL Annealing)
│   │   └── train_transformer.py# Training loop for Task 3
│   ├── evaluation/
│   │   ├── metrics.py          # Master evaluation script
│   │   ├── pitch_histogram.py  # Pitch Histogram Similarity calculation
│   │   └── rhythm_score.py     # Rhythm Diversity Score calculation
│   └── generation/
│       ├── sample_latent.py    # Code for sampling z ~ N(0,I) for AE/VAE
│       ├── generate_music.py   # Autoregressive / decoder generation scripts
│       └── midi_export.py      # Matrix/Token to .midi file conversion
├── outputs/
│   ├── generated_midis/        # Final 5-15 MIDI samples for submission
│   ├── plots/                  # Loss curves and Perplexity plots
│   └── survey_results/         # Human listening survey datasets (RLHF)
└── report/
    ├── final_report.tex        # LaTeX source for the final PDF
    ├── architecture_diagrams/  # Model diagrams
    └── references.bib          # Citations
```

## ⚙️ Environment Setup
As noted in the Supplementary Implementation Guide, PyTorch must be installed first depending on your system's CUDA availability. 

1. **Install PyTorch:** Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for your specific OS/Compute platform command.
2. **Install Remaining Dependencies:**
```bash
pip install -r requirements.txt
```
*(Dependencies include: `pretty_midi`, `miditok`, `numpy`, `pandas`, `matplotlib`, and optionally `music21`)*

## 💾 Dataset Preparation
This project utilizes the **MAESTRO Dataset v3.0.0** (or optionally the Lakh/Groove MIDI datasets). 
1. Download the MIDI-only zip file from the [Google Magenta MAESTRO page](https://magenta.tensorflow.org/datasets/maestro).
2. Extract the `.csv` and subdirectories into `data/raw_midi/`.
3. Run the preprocessing scripts to filter sparse windows and generate piano-rolls/tokens:
```bash
python src/preprocessing/piano_roll.py  # For Tasks 1 & 2
python src/preprocessing/tokenizer.py   # For Task 3
```

## 🚀 How to Run

### 1. Training the Models
To train the individual models, navigate to the root directory and execute the respective training scripts. *Ensure hyperparameters are set in `src/config.py`.*

```bash
python src/training/train_ae.py           # Trains the LSTM Autoencoder
python src/training/train_vae.py          # Trains the VAE with KL Annealing
python src/training/train_transformer.py  # Trains the Autoregressive Transformer
```

### 2. Generating Music
To generate new MIDI compositions from the trained models:
```bash
python src/generation/generate_music.py --model vae --num_samples 8
python src/generation/generate_music.py --model transformer --num_samples 10
```
Generated `.mid` files will be saved in `outputs/generated_midis/`.

### 3. Evaluation & Metrics
To calculate Pitch Histogram Similarity, Rhythm Diversity, and Repetition Ratios against the baseline models (Random Note Generator & Markov Chain):
```bash
python src/evaluation/metrics.py --evaluate_all
```

## 📊 Evaluation Metrics & Baselines

| Model | Loss | Perplexity | Rhythm Diversity | Human Score | Genre Control |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Generator | - | - | 0.020 | 1.0 | None |
| Markov Chain | - | - | 0.130 | 2.5 | None |
| **Task 1: Autoencoder** | 0.378 | - | 0.430 | 3.2 | None |
| **Task 2: VAE** | 3109.75 | - | 0.594 | 3.8 | Basic |
| **Task 3: Transformer** | 1.850 | 6.36 | 0.021 | 4.1 | Moderate |
