import math
import numpy as np
import pandas as pd
import pretty_midi
from .pitch_histogram import pitch_histogram_similarity
from .rhythm_score import rhythm_diversity_score


def repetition_ratio(midi_path, n=4):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted([n for inst in pm.instruments for n in inst.notes], key=lambda x: x.start)
    if len(notes) <= n:
        return 0.0
    seq = [n.pitch for n in notes]
    ngrams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
    total = len(ngrams)
    counts = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1
    unique_ngrams = len(counts)
    return 1.0 - float(unique_ngrams / max(total, 1))


def perplexity_from_avg_loss(avg_loss):
    return float(math.exp(avg_loss))


def human_listening_score(csv_path, score_col="score"):
    df = pd.read_csv(csv_path)
    if score_col not in df.columns:
        raise ValueError("score column missing")
    return float(df[score_col].mean())


def evaluate_pair(real_midi_path, gen_midi_path):
    return {
        "pitch_hist": pitch_histogram_similarity(real_midi_path, gen_midi_path),
        "rhythm_diversity": rhythm_diversity_score(gen_midi_path),
        "repetition_ratio": repetition_ratio(gen_midi_path),
    }
