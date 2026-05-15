from __future__ import annotations

_KEY_FLATS = {
    "f_major": {"a#": "Bb"},
    "bb_major": {"a#": "Bb", "d#": "Eb"},
    "eb_major": {"a#": "Bb", "d#": "Eb", "g#": "Ab"},
    "ab_major": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db"},
    "db_major": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db", "f#": "Gb"},
    "gb_major": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db", "f#": "Gb"},
    "d_minor": {"a#": "Bb"},
    "g_minor": {"a#": "Bb", "d#": "Eb"},
    "c_minor": {"a#": "Bb", "d#": "Eb", "g#": "Ab"},
    "f_minor": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db"},
    "bb_minor": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db", "f#": "Gb"},
    "eb_minor": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db", "f#": "Gb"},
}


def spell_pitch_name(name: str, key: str | None) -> str:
    """Re-spell a sharp-style pitch name, e.g. A#4 -> Bb4 in flat keys."""

    if not name or not key or "#" not in name:
        return name
    key_norm = key.strip().lower().replace(" ", "_").replace("-", "_")
    flats = _KEY_FLATS.get(key_norm)
    if not flats:
        return name
    pc = ""
    octave = ""
    for ch in name:
        if ch.isdigit() or ch == "-":
            octave += ch
        else:
            pc += ch
    flat = flats.get(pc.lower())
    return (flat + octave) if flat else name


def spell_pitch_names(names, key: str | None):
    if names is None:
        return names
    return [spell_pitch_name(str(name), key) for name in names]
