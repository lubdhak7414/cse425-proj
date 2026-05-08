import numpy as np
import pretty_midi


def pitch_class_histogram(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    counts = np.zeros(12, dtype=np.float32)
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return counts
    for note in notes:
        counts[note.pitch % 12] += 1.0
    return counts / max(counts.sum(), 1.0)


def pitch_histogram_similarity(real_midi_path, gen_midi_path):
    p = pitch_class_histogram(real_midi_path)
    q = pitch_class_histogram(gen_midi_path)
    return float(np.abs(p - q).sum())
