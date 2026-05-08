import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from miditok import REMI, TokenizerConfig

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from generation.midi_export import validate_midi
from src.models.transformer import build_transformer


def sample_next_token(logits, temperature=1.0, top_k=20):
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        return indices[torch.multinomial(probs, 1)].item()
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_sequence(model, genre_id, max_len, temperature, top_k, device, bos_token, vocab_size):
    tokens = [bos_token] if bos_token is not None else [np.random.randint(0, vocab_size)]
    model.eval()
    for _ in range(max_len - 1):
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        genre = torch.tensor([genre_id], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x, genre)[:, -1, :].squeeze(0)
        tokens.append(sample_next_token(logits, temperature=temperature, top_k=top_k))
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI with a trained transformer.")
    parser.add_argument("--checkpoint", type=str, default=str(repo_root / "models" / "saved" / "transformer.pth"))
    parser.add_argument("--out-dir", type=str, default=str(repo_root / "outputs" / "generated_midis" / "task3"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-velocities", type=int, default=32)
    args = parser.parse_args()

    config = TokenizerConfig(
        num_velocities=args.num_velocities,
        use_chords=False,
        use_programs=False,
        one_token_stream=True,
    )
    tokenizer = REMI(config)
    tokenizer.one_token_stream = True
    vocab_size = tokenizer.vocab_size

    genre_path = repo_root / "data" / "processed" / "tokens" / "genres.npy"
    legacy_genre = repo_root / "data" / "processed_tokens" / "genres.npy"
    if not genre_path.exists():
        genre_path = legacy_genre
    if genre_path.exists():
        genres = np.load(genre_path)
        genre_count = int(np.max(genres)) + 1 if len(genres) else 1
    else:
        genre_count = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer(vocab_size=vocab_size, genre_count=genre_count).to(device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    bos_token = None
    try:
        bos_token = tokenizer["BOS_None"]
    except Exception:
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for i in range(args.num_samples):
        genre_id = i % max(genre_count, 1)
        tokens = generate_sequence(
            model,
            genre_id,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            bos_token=bos_token,
            vocab_size=vocab_size,
        )
        pm = tokenizer.tokens_to_midi([tokens])
        if pm is None:
            continue
        out_path = out_dir / f"genre_{genre_id}_sample_{i + 1}.mid"
        pm.write(str(out_path))
        if validate_midi(str(out_path)):
            generated += 1
        else:
            out_path.unlink(missing_ok=True)
    print(f"Generated {generated} MIDI files.")


if __name__ == "__main__":
    main()
