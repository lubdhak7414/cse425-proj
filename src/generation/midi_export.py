import pretty_midi


def write_notes_to_midi(notes, out_path, program=0):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program)
    for pitch, start, end, velocity in notes:
        inst.notes.append(pretty_midi.Note(int(velocity), int(pitch), float(start), float(end)))
    pm.instruments.append(inst)
    pm.write(out_path)


def piano_roll_to_midi(proll, out_path, fs=16, pitch_start=21, velocity=80):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    if proll.shape[0] == 88 and proll.shape[1] != 88:
        proll = proll.T
    time_steps, pitches = proll.shape
    for p in range(pitches):
        active = False
        start = 0
        for t in range(time_steps):
            if proll[t, p] == 1 and not active:
                active = True
                start = t / fs
            elif proll[t, p] == 0 and active:
                active = False
                inst.notes.append(pretty_midi.Note(velocity, p + pitch_start, start, t / fs))
        if active:
            inst.notes.append(pretty_midi.Note(velocity, p + pitch_start, start, time_steps / fs))
    pm.instruments.append(inst)
    pm.write(out_path)


def validate_midi(midi_path, min_notes=50, min_seconds=5.0):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return False
    notes = [n for inst in pm.instruments for n in inst.notes]
    if len(notes) < min_notes:
        return False
    return pm.get_end_time() >= min_seconds
