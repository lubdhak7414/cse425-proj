import argparse
import random
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from generation.midi_export import write_notes_to_midi, validate_midi
from src.preprocessing.midi_parser import list_midi_files
import pretty_midi


PITCH_RANGE = (21, 109)
DURATION_CHOICES = [0.25, 0.5, 0.75, 1.0]
VELOCITY_RANGE = (60, 100)


def generate_random(notes=200, total_seconds=30.0):
    notes_out = []
    for _ in range(notes):
        pitch = random.randint(PITCH_RANGE[0], PITCH_RANGE[1] - 1)
        duration = random.choice(DURATION_CHOICES)
        start = random.uniform(0, max(total_seconds - duration, 0.1))
        end = start + duration
        velocity = random.randint(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
        notes_out.append((pitch, start, end, velocity))
    return notes_out


def extract_pitch_sequence(midi_path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted([n for inst in pm.instruments for n in inst.notes], key=lambda x: x.start)
    return [n.pitch for n in notes]


def build_markov_transition(midi_files):
    counts = {}
    for midi_path in midi_files:
        try:
            seq = extract_pitch_sequence(midi_path)
        except Exception:
            continue
        for i in range(len(seq) - 1):
            counts.setdefault(seq[i], {})
            counts[seq[i]][seq[i + 1]] = counts[seq[i]].get(seq[i + 1], 0) + 1
    transition = {}
    for pitch, next_counts in counts.items():
        total = sum(next_counts.values())
        if total == 0:
            continue
        probs = {k: v / total for k, v in next_counts.items()}
        transition[pitch] = probs
    return transition


def sample_markov(transition, notes=200, total_seconds=30.0):
    if not transition:
        return generate_random(notes=notes, total_seconds=total_seconds)
    pitches = list(transition.keys())
    current = random.choice(pitches)
    notes_out = []
    start = 0.0
    for _ in range(notes):
        duration = random.choice(DURATION_CHOICES)
        end = start + duration
        velocity = random.randint(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
        notes_out.append((current, start, end, velocity))
        next_probs = transition.get(current, None)
        if next_probs:
            choices = list(next_probs.keys())
            probs = list(next_probs.values())
            current = random.choices(choices, probs)[0]
        else:
            current = random.choice(pitches)
        start = min(end, total_seconds - duration)
    return notes_out


def main():
    parser = argparse.ArgumentParser(description="Generate Random and Markov baseline MIDI samples.")
    parser.add_argument("--raw-dir", type=str, default=str(repo_root / "data" / "raw_midi" / "maestro-v3.0.0"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--notes", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default=str(repo_root / "outputs" / "generated_midis" / "baselines"))
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = list_midi_files(args.raw_dir)
    transition = build_markov_transition(midi_files[:200]) if midi_files else {}

    for i in range(args.num_samples):
        rand_path = out_dir / f"random_{i + 1}.mid"
        write_notes_to_midi(generate_random(notes=args.notes), str(rand_path))
        if not validate_midi(str(rand_path)):
            rand_path.unlink(missing_ok=True)

        markov_path = out_dir / f"markov_{i + 1}.mid"
        write_notes_to_midi(sample_markov(transition, notes=args.notes), str(markov_path))
        if not validate_midi(str(markov_path)):
            markov_path.unlink(missing_ok=True)

    print(f"Baseline MIDI files written to {out_dir}")


if __name__ == "__main__":
    main()
