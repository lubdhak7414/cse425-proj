from miditok import REMI, TokenizerConfig


def build_tokenizer(num_velocities=32):
    config = TokenizerConfig(num_velocities=num_velocities, use_chords=False, use_programs=False)
    return REMI(config)


def midi_to_tokens(tokenizer, midi_path, min_len=0):
    tokens = tokenizer(midi_path)
    if len(tokens.ids) < min_len:
        return None
    return tokens.ids
