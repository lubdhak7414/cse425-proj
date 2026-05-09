# This file has been worked on by Safwan Usaid Lubdhak
from pathlib import Path
import pretty_midi


def load_midi(midi_path: str | Path) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(midi_path))


def list_midi_files(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    files = list(root.rglob("*.mid")) + list(root.rglob("*.midi"))
    return sorted(files)
