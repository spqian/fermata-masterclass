from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InstrumentProfile:
    id: str
    instrument: str
    family: str
    pitch_class: str
    intonation: dict[str, Any] = field(default_factory=dict)
    teacher_examples: str = "your favorite great teacher"
    voicing_focus: str = ""
    category_guidance: str = ""
    measurements_available: str = ""
    measurable_claims_rule: str = ""
    intonation_or_voicing: str = "intonation"
    video_checklist: str = ""
    disabled_tools: tuple[str, ...] = ()
    polyphony: str = "low"
    max_simultaneous_voices: int = 2
    voicing: dict[str, Any] = field(default_factory=dict)
    rhythm: dict[str, Any] = field(default_factory=dict)
    hmm: dict[str, Any] = field(default_factory=dict)
    onset_detection: dict[str, Any] = field(default_factory=dict)
    comment_generator: dict[str, Any] = field(default_factory=dict)


PIANO_VOICING_FOCUS = (
    "For piano, voicing IS the technique — which finger projects the melody over which "
    "supporting voices, how chordal balance shapes harmony, how pedal use blends or "
    "separates voices. Use inspect_voicing for chord-member evidence and measure_dynamics "
    "for phrase-level dynamic shape."
)

PIANO_CATEGORY_GUIDANCE = """- **musical** (≥3 per take): name the harmonic crux, the phrase shape, the form, the period style.
- **voicing** (≥2 per take, tool-backed): which voice carries melody, how chordal balance shapes the harmony. Use inspect_voicing for chord balance/top-voice projection/attack spread/pedal residue; use measure_dynamics for longer phrase swells.
- **technique**: touch, pedaling (audio + visual), articulation. Hypothesize from sound; mark visual claims with the side-camera caveat.
- **rhythm/tempo**: rubato, agogic accents, tempo plan across the take.
- **camera / setup**: only if a visible problem exists."""

PIANO_MEASURABLE_CLAIMS_RULE = (
    "If you say 'the top voice is buried', call inspect_voicing first. "
    "If you say 'the phrase crescendo peaks too early', call measure_dynamics first. "
    "If you say 'you stretched bar 5 by 30%', call measure_tempo first. "
    "If you say 'the trill is at 8 Hz', call measure_trill first. "
    "Piano has fixed pitch — DO NOT discuss intonation. Skip the inspect_intonation tool."
)

PIANO_VIDEO_CHECKLIST = """For piano, look in the frames for:
- **hand & wrist**: wrist height relative to keys (collapsed wrist = lost projection); flat vs. curved fingers; fingertip vs. pad contact; thumb position on chords.
- **fingering choices**: visible fingers landing on the keys at moments the audio shows trouble (wrong notes, late attacks, broken phrasing). If you can SEE the finger that just played, name it.
- **arm & shoulder**: free elbow / dropped shoulder vs. tension; lateral arm motion for wide leaps; whole-arm weight on accented chords.
- **posture**: bench height (forearm parallel to keys?), spine alignment, sit-bone contact; head/neck tension during difficult passages.
- **pedaling**: right foot — pedal changes per bar, half-pedaling, late releases that you can hear muddying harmonies. Una-corda foot if visible.
- **gesture & intent**: breathing/sway at phrase starts and endings; visible rubato setup; whether the body anticipates the next gesture.

To judge **motion** (pedal change timing, finger transitions, arm drop on accents), don't rely on the sample frames — call `get_frames(start_sec=X, end_sec=X+1.5, fps=4)` to get a 6-frame burst around the moment in question. Read the burst as motion: where is the pedal foot in frame 1 vs. frame 6? Did the wrist drop after the chord attack?

If the camera angle hides hands or pedal entirely (common with phone-on-music-stand), say so plainly in `next_take` and `lesson.next_take` (e.g., \"Set the camera at 45° from the right of the keyboard so we can see both hands and the pedal foot.\")."""

BOWED_STRING_VIDEO_CHECKLIST = """For violin/viola/cello, look in the frames for:
- **bow**: contact point (sul tasto vs. ponticello vs. middle), bow division (where in the bow you're playing — frog/middle/tip), bow tilt/hair coverage, straightness across the string.
- **bow speed and pressure**: a SINGLE frame can't tell you bow speed — you need a burst. Call `get_frames(start_sec=X, end_sec=X+0.8, fps=6)` to get ~5 frames across <1s. Compare bow tip position frame-to-frame: large displacement = fast bow, small = slow. Combine with the sound (loud + slow bow = lots of pressure, soft + fast bow = light/airy).
- **bow arm**: elbow height relative to the string being played (cello: elbow level with string; violin: shoulder relaxed, elbow leads on up-bows); wrist suppleness at bow changes vs. locked-up wrist that breaks the line.
- **left hand**: thumb position (over the neck for violin/viola, behind the neck for cello), finger curvature, vibrato (if you see it as wrist or arm motion in a burst), shifts (preparation before vs. after the shift).
- **instrument hold**: scroll height (drooping = collapse), chinrest contact for violin/viola (no pinching), endpin angle and chest contact for cello.
- **posture**: shoulders, neck, breathing, eye contact with the score; whether the body absorbs the bow weight.

To assess motion-based things — bow speed, vibrato rate, shift speed, bow change articulation — extract a short burst with `get_frames(start_sec=X, end_sec=X+1, fps=4-6)`. Six frames in 1 second is enough to track most bow gestures.

If the camera angle hides the bow or the left hand, name the missing angle in `next_take` (e.g., \"Re-record from 30° in front and slightly to your left so we can see bow contact-point and the left hand together.\")."""

DEFAULT_VIDEO_CHECKLIST = """Look in the frames for posture, breathing, hand/finger position, and any visible technique cue (embouchure, breath support, body stance). For motion-based observations (key/valve velocity, breath pacing, articulation gesture), call `get_frames(start_sec=X, end_sec=X+1, fps=4)` to extract a short burst — a single frame can't show motion. If the camera angle hides what would be most useful, name the missing angle in `next_take`."""

BOWED_STRING_VOICING_FOCUS = (
    "For violin/viola/cello, comment on voicing within chords (which string carries the "
    "melody) AND across single-line phrases (the long-arc dynamic shape and which notes "
    "are the structural pillars)."
)

BOWED_STRING_CATEGORY_GUIDANCE = """- **musical** (≥3 per take): name the harmonic crux, the phrase shape, the form, the period style.
- **technique** (audio + visual where possible): bow speed, contact-point, ornament execution. "From the frames around 60s, your bow is near the tip on the chord — that limits dynamic range. Try saving more bow earlier in the bar."
- **intonation**: tool-backed only. "Your A-naturals spread 61 cents across the take (verified via inspect_intonation on bars 5 and 6); pick a single A and drone-tune."
- **voicing within phrases** (tool-backed): use measure_dynamics to identify which notes the player emphasized vs. swallowed in the melodic line. "In bar 3 the descending sixteenths peak at the second note (-2dB) rather than the first (-7dB), which inverts the natural melodic shape."
- **camera / setup**: only if a visible problem exists. Mark with "verify with side-angle camera" caveats."""

BOWED_STRING_MEASURABLE_CLAIMS_RULE = (
    "If you say 'you played that A 30 cents sharp', you must call inspect_intonation first. "
    "If you say 'the trill is at 8 Hz', you must call measure_trill. "
    "If you say 'you stretched bar 8 by 40%', you must call measure_tempo. "
    "If you say 'the top voice is buried in this phrase', you must call measure_dynamics."
)

DEFAULT_VOICING_FOCUS = (
    "Comment on voicing within chords/textures and the long-arc dynamic shape of each "
    "phrase. Identify which notes are structural pillars and which are passing."
)

DEFAULT_CATEGORY_GUIDANCE = """- **musical** (≥3 per take): name the harmonic crux, the phrase shape, the form, the period style.
- **technique**: posture, sound production, articulation.
- **intonation**: tool-backed only for variable-pitch instruments.
- **rhythm/tempo**: pulse, rubato, structural pacing.
- **dynamics/voicing**: long-arc dynamic shape and balance within textures."""

DEFAULT_MEASURABLE_CLAIMS_RULE = (
    "Ground measurable claims with tools: use inspect_intonation for pitch claims when available, "
    "measure_tempo for tempo claims, measure_trill for trill claims, and measure_dynamics or "
    "inspect_voicing for dynamics and voicing claims."
)






BUILTIN_PROFILES: dict[str, InstrumentProfile] = {
    "piano": InstrumentProfile(
        id="piano",
        instrument="piano",
        family="keyboard",
        pitch_class="fixed",
        polyphony="high",
        max_simultaneous_voices=10,
        intonation={"enabled": False},
        voicing={"enabled": True},
        rhythm={"min_onset_spacing_ms": 25},
        hmm={"search_bins": 0, "harmonic_weight": 0.5, "hesitation_factor": 1.5, "skip_penalty": 0.01},
        onset_detection={"peak_delta": 0.10, "min_spacing_ms": 30},
        comment_generator={"voicing_threshold_db": [3, 6, 10], "rhythm_attack_threshold_ms": [40, 80, 150]},
        teacher_examples="András Schiff, Mitsuko Uchida, Daniil Trifonov",
        voicing_focus=PIANO_VOICING_FOCUS,
        category_guidance=PIANO_CATEGORY_GUIDANCE,
        measurements_available="voicing energy, rhythm/onset timing, tempo, HMM alignment",
        measurable_claims_rule=PIANO_MEASURABLE_CLAIMS_RULE,
        intonation_or_voicing="voicing",
        video_checklist=PIANO_VIDEO_CHECKLIST,
        disabled_tools=("inspect_intonation",),
    ),
    "violin_solo": InstrumentProfile(
        id="violin_solo",
        instrument="violin",
        family="bowed_string",
        pitch_class="variable",
        polyphony="low",
        max_simultaneous_voices=4,
        intonation={"enabled": True, "search_cents": 50, "temperaments": ["12_tet", "just_b_minor", "pythagorean"]},
        rhythm={"min_onset_spacing_ms": 80},
        hmm={"search_bins": 2, "harmonic_weight": 0.3, "hesitation_factor": 2.5, "skip_penalty": 0.005},
        onset_detection={"peak_delta": 0.07, "min_spacing_ms": 80},
        comment_generator={"intonation_threshold_cents": [15, 25], "rubato_threshold_pct": [15, 30, 50]},
        teacher_examples="Itzhak Perlman, Rachel Barton Pine, Augustin Hadelich",
        voicing_focus=BOWED_STRING_VOICING_FOCUS,
        category_guidance=BOWED_STRING_CATEGORY_GUIDANCE,
        measurements_available="intonation cents, HMM alignment, onsets, bar timing, dynamics envelope",
        measurable_claims_rule=BOWED_STRING_MEASURABLE_CLAIMS_RULE,
        video_checklist=BOWED_STRING_VIDEO_CHECKLIST,
    ),
    "viola_solo": InstrumentProfile(
        id="viola_solo",
        instrument="viola",
        family="bowed_string",
        pitch_class="variable",
        polyphony="low",
        max_simultaneous_voices=4,
        intonation={"enabled": True, "search_cents": 50, "temperaments": ["12_tet", "just_d_minor", "pythagorean"]},
        rhythm={"min_onset_spacing_ms": 80},
        hmm={"search_bins": 2, "harmonic_weight": 0.3, "hesitation_factor": 2.5, "skip_penalty": 0.005},
        onset_detection={"peak_delta": 0.07, "min_spacing_ms": 80},
        comment_generator={"intonation_threshold_cents": [15, 25], "rubato_threshold_pct": [15, 30, 50]},
        teacher_examples="Tabea Zimmermann, Kim Kashkashian, Lawrence Power",
        voicing_focus=BOWED_STRING_VOICING_FOCUS,
        category_guidance=BOWED_STRING_CATEGORY_GUIDANCE,
        measurements_available="intonation cents, HMM alignment, onsets, bar timing, dynamics envelope",
        measurable_claims_rule=BOWED_STRING_MEASURABLE_CLAIMS_RULE,
        video_checklist=BOWED_STRING_VIDEO_CHECKLIST,
    ),
    "cello_solo": InstrumentProfile(
        id="cello_solo",
        instrument="cello",
        family="bowed_string",
        pitch_class="variable",
        polyphony="low",
        max_simultaneous_voices=4,
        intonation={"enabled": True, "search_cents": 50, "temperaments": ["12_tet", "just_g_major", "pythagorean"]},
        rhythm={"min_onset_spacing_ms": 80},
        hmm={"search_bins": 2, "harmonic_weight": 0.3, "hesitation_factor": 2.5, "skip_penalty": 0.005},
        onset_detection={"peak_delta": 0.07, "min_spacing_ms": 80},
        comment_generator={"intonation_threshold_cents": [15, 25], "rubato_threshold_pct": [15, 30, 50]},
        teacher_examples="Yo-Yo Ma, Steven Isserlis, Mischa Maisky",
        voicing_focus=BOWED_STRING_VOICING_FOCUS,
        category_guidance=BOWED_STRING_CATEGORY_GUIDANCE,
        measurements_available="intonation cents, HMM alignment, onsets, bar timing, dynamics envelope",
        measurable_claims_rule=BOWED_STRING_MEASURABLE_CLAIMS_RULE,
        video_checklist=BOWED_STRING_VIDEO_CHECKLIST,
    ),
    "default": InstrumentProfile(
        id="default",
        instrument="musician",
        family="unknown",
        pitch_class="variable",
        polyphony="low",
        max_simultaneous_voices=2,
        intonation={"enabled": True, "search_cents": 50, "temperaments": ["12_tet"]},
        rhythm={"min_onset_spacing_ms": 80},
        hmm={"search_bins": 2, "harmonic_weight": 0.3, "hesitation_factor": 2.5, "skip_penalty": 0.005},
        onset_detection={"peak_delta": 0.07, "min_spacing_ms": 80},
        comment_generator={"intonation_threshold_cents": [15, 25], "rubato_threshold_pct": [15, 30, 50]},
        teacher_examples="your favorite great teacher",
        voicing_focus=DEFAULT_VOICING_FOCUS,
        category_guidance=DEFAULT_CATEGORY_GUIDANCE,
        measurements_available="intonation cents when available, rhythm/onset timing, tempo, dynamics envelope, HMM alignment",
        measurable_claims_rule=DEFAULT_MEASURABLE_CLAIMS_RULE,
    ),
}

_ALIASES = {
    "piano_solo": "piano",
    "piano_with_pedal": "piano",
}


def load_instrument_profile(profile_id: str | None) -> InstrumentProfile:
    """Return a built-in instrument profile, falling back to the default profile."""
    key = (profile_id or "").strip().lower()
    key = _ALIASES.get(key, key)
    return BUILTIN_PROFILES.get(key) or BUILTIN_PROFILES["default"]


def intonation_enabled_for_profile(profile: InstrumentProfile) -> bool:
    """Whether intonation-specific analysis and tools should be enabled for this profile."""
    return bool(profile.intonation.get("enabled", True))


def system_instruction_for_profile(profile: InstrumentProfile, template: str | None = None, *, tool_catalog: str | None = None) -> str:
    """Fill a teach-system-instruction template from an instrument profile."""
    if template is None:
        from masterclass.prompts import load_teacher_prompt
        template = load_teacher_prompt()
    intonation_or_voicing = profile.intonation_or_voicing or (
        "intonation" if intonation_enabled_for_profile(profile) else "voicing"
    )
    instruction = template.format(
        instrument=profile.instrument,
        teacher_examples=profile.teacher_examples,
        voicing_focus=profile.voicing_focus,
        category_guidance=profile.category_guidance,
        intonation_or_voicing=intonation_or_voicing,
        measurements_available=profile.measurements_available,
        measurable_claims_rule=profile.measurable_claims_rule,
        video_checklist=profile.video_checklist or DEFAULT_VIDEO_CHECKLIST,
    )
    if tool_catalog:
        instruction = instruction.rstrip() + "\n\n# Available investigation tools\n\n" + tool_catalog.strip() + "\n"
    return instruction
