import numpy as np
import pretty_midi


def rhythm_diversity_score(midi_path, quant_ms=50):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = [n for inst in pm.instruments for n in inst.notes]
    if not notes:
        return 0.0
    durations = np.array([n.end - n.start for n in notes], dtype=np.float32)
    q = max(quant_ms / 1000.0, 1e-6)
    quantized = np.round(durations / q) * q
    unique_count = len(np.unique(quantized))
    return float(unique_count / len(quantized))
