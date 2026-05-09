# This file has been worked on by Safwan Usaid Lubdhak
from miditok import REMI, TokenizerConfig


def build_tokenizer(num_velocities=32):
    config = TokenizerConfig(
        num_velocities=num_velocities,
        use_chords=False,
        use_programs=False,
        one_token_stream=True,
    )
    return REMI(config)


def normalize_token_ids(tokens):
    if hasattr(tokens, "ids"):
        ids = tokens.ids
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return ids
        return [ids]
    if isinstance(tokens, list):
        out = []
        for t in tokens:
            if hasattr(t, "ids"):
                ids = t.ids
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    out.extend(ids)
                else:
                    out.append(ids)
        return out
    return []


def midi_to_tokens(tokenizer, midi_path, min_len=0):
    tokens = tokenizer(midi_path)
    sequences = normalize_token_ids(tokens)
    if not sequences:
        return None
    for seq in sequences:
        if len(seq) >= min_len:
            return seq
    return None
