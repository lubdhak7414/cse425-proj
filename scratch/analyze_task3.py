import pretty_midi
from pathlib import Path

def analyze_midis(directory):
    path = Path(directory)
    files = sorted(list(path.glob("*.mid")))
    print(f"{'Filename':<25} | {'Notes':<5} | {'Duration':<10}")
    print("-" * 45)
    for f in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(f))
            notes = sum(len(inst.notes) for inst in pm.instruments)
            duration = pm.get_end_time()
            print(f"{f.name:<25} | {notes:<5} | {duration:.2f}s")
        except Exception as e:
            print(f"{f.name:<25} | Error: {e}")

if __name__ == "__main__":
    analyze_midis("/Users/maidulislam/untitled folder 5/cse425-proj/outputs/generated_midis/task3")
