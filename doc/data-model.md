# Data Model

Reference for every persistent JSON shape in v2. Useful when reading storage by hand, building tools that consume artifacts, or debugging shape drift.

## Manifests

### `MasterclassManifest` (`masterclass.json`)

One per class series. Lives at `tenant/{t}/users/{u}/masterclasses/{mid}/masterclass.json`.

```json
{
  "schema_version": 1,
  "masterclass": {
    "tenant_id": "pqian",
    "user_id": "pqian",
    "masterclass_id": "7db837bb..."
  },
  "piece_name": "J. S. Bach - Violin Sonata No. 1 in G minor",
  "movement": "Adagio",
  "work_id": null,
  "instrument": "violin",
  "instrument_profile": "violin_solo",
  "lessons": ["c1bd8abe...", "a933ec4b..."],
  "artifacts": {
    "reference/score_pdf": "tenant/.../reference/score_pdf",
    "reference/midi": "tenant/.../reference/midi",
    "reference/score_pages/page-001.png": "...",
    ...
  },
  "metadata": {
    "score_prep_state": "ready",
    "score_prep_source": "audiveris",
    "score_prep_substage": null,
    "score_prep_started_at": "...",
    "score_prep_updated_at": "...",
    "score_prep_elapsed_sec": 91.6,
    "score_prep_layout_measure_count": 198,
    "score_prep_played_movement_measure_count": 22,
    "score_prep_midi_measure_count": 22,
    "score_prep_redistributed_to_midi": true,
    "score_prep_barline_corrections": 0,
    "played_movement_id": 1,
    "midi_find_state": "ready",
    "midi_find_substage": null,
    "reference_midi_filename": "auto.mid",
    "reference_midi_url": "https://www.mutopiaproject.org/.../bwv-1001_1.mid",
    "reference_midi_source": "mutopia",
    "reference_midi_attribution": "J. S. Bach (1685–1750) — BWV 1001 Adagio (via mutopia)",
    "reference_midi_confidence": "high"
  },
  "created_at": "...",
  "updated_at": "..."
}
```

### `SessionManifest` (`session.json`)

One per lesson upload. Lives at `tenant/{t}/users/{u}/sessions/{sid}/session.json`.

```json
{
  "schema_version": 1,
  "session": {
    "tenant_id": "pqian",
    "user_id": "pqian",
    "session_id": "a933ec4b..."
  },
  "repertoire": "J. S. Bach - Violin Sonata No. 1 in G minor",
  "movement": "Adagio",
  "instrument": "violin",
  "instrument_profile": "violin_solo",
  "notes": "My first take on the first 9 bars of this piece",
  "state": "ready",
  "artifacts": {
    "input/source_video": "tenant/.../input/Partita.mp4",
    "artifacts/audio.wav": "...",
    "artifacts/audio_16k.wav": "...",
    "artifacts/frames": "...",
    "score/score_map.json": "...",
    "context/prior_lessons.json": "...",
    ...
  },
  "metadata": {
    "masterclass_id": "7db837bb...",
    "first_measure": null,
    "last_measure": null,
    "auto_detected_first_measure": 1,
    "auto_detected_last_measure": 9,
    "extract_media_state": "ready",
    "analyze_state": "ready",
    "evidence_packet_state": "ready",
    "onsets_state": "ready",
    "hmm_align_state": "ready",
    "score_map_state": "ready",
    "intonation_state": "ready",
    "rhythm_state": "ready",
    "voicing_state": "skipped",
    "mechanical_comments_state": "ready",
    "teach_state": "ready",
    "frames": ["tenant/.../artifacts/frames/frame_0001.jpg", ...]
  },
  "errors": [],
  "created_at": "...",
  "updated_at": "..."
}
```

State transitions: `created → uploaded → processing → ready | failed`. The 12 stage states are independent — each can be `queued | running | ready | failed | skipped`.

## Engine artifacts

### `score_prep.json` (per masterclass)

```json
{
  "first_music_page": 1,
  "page_count": 7,
  "instrument": "violin",
  "movements": [
    {
      "id": 1,
      "title": "Adagio",
      "tempo_marking": "Adagio",
      "time_signature": "4/4",
      "key_signature": "G minor",
      "start_page": 1,
      "end_page": 1,
      "first_measure": 1,
      "last_measure": 22,
      "measure_count": 22
    },
    ...
  ],
  "pages": [
    {
      "page": 1,
      "kind": "music",
      "movement_id": 1,
      "first_measure": 1,
      "last_measure": 22,
      "system_count": 10,
      "systems": [
        {
          "system_index": 1,
          "first_measure": 1,
          "last_measure": 2,
          "bbox": {"x": 0.094, "y": 0.139, "w": 0.842, "h": 0.039},
          "music_start_x_frac": 0.098,
          "bars": [
            {"bar_number": 1, "x_frac_start": 0.099, "x_frac_end": 0.556},
            {"bar_number": 2, "x_frac_start": 0.556, "x_frac_end": 0.940}
          ]
        },
        ...
      ]
    },
    ...
  ],
  "_meta": {
    "model": "gemini-2.5-pro",
    "page_dpi": 150,
    "rasterized_page_count": 7,
    "generated_at": "...",
    "source": "audiveris",
    "audiveris_version": "5.6.2",
    "audiveris_time_sec": 87.5,
    "redistributed_movement_to_midi": {
      "movement_id": 1,
      "midi_bar_count": 22,
      "system_counts": [2, 3, 2, 3, 2, 2, 2, 2, 2, 3]
    }
  }
}
```

`bbox` and `bars[].x_frac_*` are normalized [0, 1] over the page raster. Multiply by image natural dimensions to get pixel coordinates.

### `score_map.json` (per session)

Combines masterclass `score_prep.json` + reference MIDI + session HMM alignment.

```json
{
  "schema_version": 1,
  "alignment_source": "hmm",
  "instrument": "violin",
  "key": "G minor",
  "movement": "Adagio",
  "masterclass_id": "7db837bb...",
  "score_id": "...",
  "first_measure": 1,
  "last_measure": 9,
  "total_measures": 9,
  "played_lo": 1,
  "played_hi": 9,
  "systems": [
    {
      "system": 101,
      "page": 1,
      "system_on_page": 1,
      "first_measure": 1,
      "last_measure": 2,
      "bbox": {"x": 0.094, "y": 0.139, "w": 0.842, "h": 0.039},
      "bars": [1, 2],
      "first_bar_x_pad_frac": 0.06,
      "trailing_x_pad_frac": 0.04,
      "image": "score/page-001.png",
      "image_kind": "page"
    },
    ...
  ],
  "bars": [
    {
      "measure": 1,
      "midi_measure": 1,
      "page": 1,
      "system": 101,
      "system_on_page": 1,
      "image": "score/page-001.png",
      "bbox": {"x": 0.099, "y": 0.139, "w": 0.457, "h": 0.039},
      "system_bbox": {"x": 0.094, "y": 0.139, "w": 0.842, "h": 0.039},
      "highlight_x_frac": [0.0, 0.547],
      "alignment_source": "score_prep_bars"
    },
    ...
  ],
  "notes": [
    {
      "note_id": "m1_b1.00_G5+Bb4+D4+G3",
      "system": 101,
      "measure": 1,
      "midi_measure": 1,
      "beat_in_bar": 1.0,
      "score_time": 0.0,
      "perf_time": 4.899,
      "x_frac": 0.0,
      "x_frac_end": 0.018,
      "names": ["G5", "Bb4", "D4", "G3"],
      "pitch_midi": [79, 70, 62, 55],
      "midi_pitches": [79, 70, 62, 55],
      "is_chord": true,
      "hmm_confidence": "high",
      "hmm_dwell_sec": 0.42,
      "interpolated": false,
      "is_bar_anchor": true,
      "confidence": "high"
    },
    ...
  ],
  "notes_help": "...",
  "_meta": {...}
}
```

System ID convention: `100 * page + system_on_page`. So page 1 system 1 = 101, page 6 system 2 = 602.

Note ID convention: `m{measure}_b{beat:.2f}_{pitch_names_joined_with_+}`. Stable across re-alignments.

### `hmm_alignment.json` (per session)

```json
{
  "measure_timestamps": [
    {"measure": 1, "start": 4.899},
    {"measure": 2, "start": 13.955},
    ...
  ],
  "bar_starts": [
    {
      "measure": 1,
      "performed_time_sec": 4.899,
      "first_visited_pitches": ["(music start)"],
      "is_score_bar_first_state": true,
      "method": "music_start"
    },
    {
      "measure": 2,
      "performed_time_sec": 13.955,
      "first_visited_pitches": ["F5"],
      "is_score_bar_first_state": true,
      "method": "global_dp",
      "loudness_db": -18.2,
      "expected_t_sec": 12.34,
      "delta_from_expected_ms": 1615.0
    },
    ...
  ],
  "measure_count": 9,
  "audio_total_seconds": 132.07,
  "midi_total_seconds": 88.0,
  "summary": {
    "method": "hmm_viterbi+onset_refine+global_dp",
    "note_count": 416,
    "state_count": 142,
    "state_coverage": 0.987,
    "refinement_applied": true,
    "notes_with_onset_correction": 399,
    "mean_onset_correction_ms": -62,
    "bars_anchored_to_onsets": 8,
    "bars_no_onset_match": 1,
    "effective_first_measure": 1,
    "effective_last_measure": 9,
    "played_range_auto_detected": true,
    "played_range_method": "auto_confidence",
    "music_start_sec": 4.9,
    "tempo_factor": 1.5
  }
}
```

### `comments_enriched.json` (teacher output)

```json
{
  "session": "a933ec4b...",
  "video_path": "...",
  "movement": "Adagio",
  "repertoire": "J. S. Bach - Violin Sonata No. 1 in G minor",
  "played_measures": [1, 9],
  "summary": "This is a much more controlled and rhythmically stable performance than your last take...",
  "progress_notes": "Compared to your previous take, the rhythmic foundation in the opening is vastly improved...",
  "enrichment_notes": "Investigated bars 1, 5, and 9 with inspect_chord, measure_dynamics, and watch...",
  "measure_timestamps": [{"measure": 1, "start": 4.9}, ...],
  "lesson": {
    "artistic_summary": "Excellent progress. You've taken the architectural concept...",
    "what_works": ["...", "..."],
    "areas_to_develop": [
      {"focus": "Opening Chord Tone and Voicing", "priority": "high", "exercise": "..."},
      ...
    ],
    "this_week_practice": ["...", "..."],
    "next_take": "..."
  },
  "comments": [
    {
      "id": "g_001",
      "start": 4.83,
      "end": 10.0,
      "category": "musical|voicing|intonation|rhythm|technique",
      "severity": "info|warn|alert",
      "summary": "Opening chord attack is still too sharp",
      "text": "1-3 sentences ending with a try-this",
      "measure": 1,
      "beat": 1.0,
      "evidence_ref": "tool:inspect_chord",
      "provenance": ["perception", "tool:measure_dynamics(start_sec=4.83,end_sec=10.0)"],
      "references": [
        {"measure": 1, "beat": 1.0, "note_name": "G5", "page": 1, "system_index": 1, "note_id": "m1_b1.00_G5+Bb4+D4+G3"}
      ],
      "note_refs": ["m1_b1.00_G5+Bb4+D4+G3"]
    },
    ...
  ],
  "dropped": [
    {"id": "c008", "reason": "single-note off-pulse outlier — below threshold"},
    ...
  ]
}
```

### `mechanical_comments.json`

```json
{
  "comments": [
    {
      "id": "c001",
      "category": "intonation",
      "severity": "info",
      "measure": 1,
      "beat": 1.0,
      "text": "G3 measured 12 cents flat",
      "evidence": {"cents": -12, "tool": "inspect_intonation"}
    },
    ...
  ],
  "summary_md": "..."
}
```

### `teach_tool_calls.json`

```json
[
  {
    "turn": 3,
    "tool": "inspect_chord",
    "status": "ok",
    "duration_sec": 0.42,
    "args": {"start_sec": 4.5, "end_sec": 6.0},
    "response_text": "Top-voice G5 ranks #5 of 8 spectral peaks..."
  },
  ...
]
```

### `prior_lessons.json` (continuity context)

```json
{
  "masterclass_id": "7db837bb...",
  "piece_name": "...",
  "movement": "Adagio",
  "work_id": null,
  "generated_at": "...",
  "lesson_count": 1,
  "lessons": [
    {
      "session_id": "c1bd8abe...",
      "created_at": "...",
      "updated_at": "...",
      "state": "ready",
      "movement": "Adagio",
      "notes": "My first take on the first 9 bars of this piece",
      "first_measure": null,
      "last_measure": 9,
      "summary": "...",
      "progress_notes": null,
      "lesson": {...},
      "teacher_comments": [...]
    }
  ]
}
```

## Schema versioning

All top-level docs include `schema_version: 1`. There has been no v2 schema change yet — when one happens, add a migration in the loader (e.g. `MasterclassStore.load_by_id` should detect `schema_version: 1` and upgrade to current).

Unwritten convention: the `_meta` fields on engine artifacts (e.g. `score_prep.json._meta`) are not subject to schema versioning — they're free-form diagnostic info added by whichever module produced the artifact.
