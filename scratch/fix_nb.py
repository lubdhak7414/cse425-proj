import json

path = "d:/cse425-proj/notebooks/baseline_markov.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if "out_dir = os.path.join('outputs', 'generated_midis', 'baselines')" in line:
                cell["source"][i] = "out_dir = os.path.join(repo_root, 'outputs', 'generated_midis', 'baselines')\n"

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
