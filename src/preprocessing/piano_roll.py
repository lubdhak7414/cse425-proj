import numpy as np
import pretty_midi


def midi_to_pianoroll(midi_path, fs=16, pitch_range=(21, 109), binarize=True):
    pm = pretty_midi.PrettyMIDI(midi_path)
    proll = pm.get_piano_roll(fs=fs)[pitch_range[0]:pitch_range[1], :]
    if binarize:
        proll = (proll > 0).astype(np.float32)
    return proll.T


def segment_pianoroll(proll, seq_len=128, sparsity_threshold=0.02):
    windows = []
    n_segments = proll.shape[0] // seq_len
    for i in range(n_segments):
        win = proll[i * seq_len : (i + 1) * seq_len, :]
        if np.mean(win) > sparsity_threshold:
            windows.append(win)
    return windows
