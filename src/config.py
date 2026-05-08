from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "train_test_split"
OUTPUTS_DIR = ROOT / "outputs"
GENERATED_DIR = OUTPUTS_DIR / "generated_midis"
PLOTS_DIR = OUTPUTS_DIR / "plots"
SURVEY_DIR = OUTPUTS_DIR / "survey_results"
