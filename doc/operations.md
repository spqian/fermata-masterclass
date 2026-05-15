# Operations

## Running the API locally

```powershell
cd C:\Users\pqian\Source\music-masterclass-v2
.\tools\python\python.exe -m uvicorn masterclass.apps.api.main:create_app --factory --host 127.0.0.1 --port 8770
```

Open `http://127.0.0.1:8770/` for the wizard. The API auto-loads `.env` from the project root on startup.

For background/detached operation, use `Start-Process` with `-RedirectStandardError` and `-RedirectStandardOutput` so logs are captured.

## First-run prerequisites

1. Set `GEMINI_API_KEY` in `.env`:
   ```
   GEMINI_API_KEY=...
   ```
2. Verify bundled Python: `tools\python\python.exe --version` should print `Python 3.12.x`
3. Verify ffmpeg: `tools\ffmpeg\bin\ffmpeg.exe -version`
4. Verify Java: `tools\jre\bin\java.exe -version` should print Temurin 21
5. Verify Audiveris: `tools\audiveris\` should exist (the `bin\` subdirectory has the launcher script)

If any of these are missing, see `tooling.md` for download URLs.

## Common tasks

### Wipe all local data

```powershell
Remove-Item C:\Users\pqian\Source\music-masterclass-v2\local_adls -Recurse -Force
New-Item -ItemType Directory C:\Users\pqian\Source\music-masterclass-v2\local_adls
```

Then restart the API.

### Re-trigger score prep on an existing masterclass

```powershell
$bid = 'MASTERCLASS_ID'
Invoke-RestMethod -Uri "http://127.0.0.1:8770/masterclasses/$bid/prepare-score" `
    -Method POST `
    -Headers @{ "X-Tenant-Id"="pqian"; "X-User-Id"="pqian" }
```

Useful after editing `engine/score_prep.py` or `engine/barline_detection.py`. Takes ~90 seconds.

### Re-find MIDI on an existing masterclass

```powershell
$bid = 'MASTERCLASS_ID'
Invoke-RestMethod -Uri "http://127.0.0.1:8770/masterclasses/$bid/find-midi" `
    -Method POST `
    -Headers @{ "X-Tenant-Id"="pqian"; "X-User-Id"="pqian" }
```

### Re-run HMM alignment on an existing lesson

```powershell
.\tools\python\python.exe scripts\rerun_hmm_align.py SESSION_ID --tenant pqian --user pqian
```

Re-runs `engine/hmm_align.py:align_lesson_with_midi_hmm` reading existing audio + MIDI + (auto-detected or user-supplied) measure range. Persists `analysis/hmm_alignment.json` + `analysis/hmm_aligned_notes.json`. Takes ~10 seconds.

### Rebuild score_map for a session

```powershell
.\tools\python\python.exe scripts\rebuild_score_map.py SESSION_ID --tenant pqian --user pqian
```

Re-joins masterclass score_prep + lesson HMM alignment + MIDI into `score/score_map.json`. Takes <1 second. Run this after `rerun_hmm_align`.

### Re-generate teacher comments

```powershell
.\tools\python\python.exe scripts\rerun_teacher.py SESSION_ID --tenant pqian --user pqian
```

Reads existing analysis artifacts and runs:
1. intonation, rhythm, voicing analyses (re-read HMM)
2. score_map rebuild
3. mechanical_comments
4. agentic teacher (Gemini Pro)

Takes 2-5 minutes. Subject to Gemini 503 throttling. Use `--skip-teacher` to just refresh deterministic outputs.

### Force-flag a stuck stage as failed

If a stage gets stuck in `running` after the process is killed:

```powershell
$path = "C:\Users\pqian\Source\music-masterclass-v2\local_adls\tenant\pqian\users\pqian\sessions\SESSION_ID\session.json"
$j = Get-Content $path -Raw | ConvertFrom-Json
$j.metadata.teach_state = "failed"
$j | ConvertTo-Json -Depth 12 | Set-Content $path -NoNewline
```

## Restart workflow

After editing engine code:

```powershell
$old = Get-Process -Name python -EA SilentlyContinue | Where-Object { $_.Path -like '*music-masterclass-v2*' }
foreach ($p in $old) { Stop-Process -Id $p.Id -Force }
Start-Sleep 2
$p = Start-Process -FilePath ".\tools\python\python.exe" `
    -ArgumentList "-m","uvicorn","masterclass.apps.api.main:create_app","--factory","--host","127.0.0.1","--port","8770" `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardError 'C:\Users\pqian\AppData\Local\Temp\v2-api.err' `
    -RedirectStandardOutput 'C:\Users\pqian\AppData\Local\Temp\v2-api.out'
Start-Sleep 6
(Invoke-WebRequest -Uri http://127.0.0.1:8770/ -UseBasicParsing).StatusCode
```

Logs at:
- `C:\Users\pqian\AppData\Local\Temp\v2-api.err` — uvicorn startup + Python errors
- `C:\Users\pqian\AppData\Local\Temp\v2-api.out` — request log + INFO

## Debugging recipes

### Inspect a masterclass's state

```powershell
$bid = 'MASTERCLASS_ID'
$j = Get-Content "C:\Users\pqian\Source\music-masterclass-v2\local_adls\tenant\pqian\users\pqian\masterclasses\$bid\masterclass.json" -Raw | ConvertFrom-Json
$j.metadata | Format-List
```

Key fields:
- `score_prep_state` (`queued | running | ready | failed | skipped`)
- `score_prep_source` (`audiveris | gemini`)
- `score_prep_substage` (live progress)
- `score_prep_layout_measure_count` vs `score_prep_midi_measure_count`
- `played_movement_id`
- `midi_find_state` + `reference_midi_url` + `reference_midi_source`

### Inspect score_prep output

```powershell
$bid = 'MASTERCLASS_ID'
$sp = Get-Content "C:\Users\pqian\Source\music-masterclass-v2\local_adls\tenant\pqian\users\pqian\masterclasses\$bid\reference\score_prep.json" -Raw | ConvertFrom-Json
"movements:"
foreach ($m in $sp.movements) {
    "  $($m.id): $($m.title) ($($m.measure_count) bars) pages $($m.start_page)-$($m.end_page)"
}
```

Per-system bar distribution for page N:

```powershell
($sp.pages | Where-Object { $_.page -eq N }).systems | ForEach-Object {
    "  sys $($_.system_index): bars $($_.first_measure)..$($_.last_measure)"
}
```

### Inspect HMM alignment

```powershell
$sess = "C:\Users\pqian\Source\music-masterclass-v2\local_adls\tenant\pqian\users\pqian\sessions\SESSION_ID"
$h = Get-Content "$sess\analysis\hmm_alignment.json" -Raw | ConvertFrom-Json
$h.summary | Format-List
$h.measure_timestamps | ForEach-Object { "  m$($_.measure): $($_.start)s" }
```

Look for:
- `audio_total_seconds` vs `bar_starts[-1].performed_time_sec` — last bar should be near audio end
- `effective_last_measure` vs MIDI bar count — was auto-detect aggressive?
- `bars_anchored_to_onsets` ratio — high = real anchors, low = fallback to even spacing
- `mean_onset_correction_ms` — magnitude of refinement

### Inspect teacher's tool calls

```powershell
$sess = "C:\Users\pqian\Source\music-masterclass-v2\local_adls\tenant\pqian\users\pqian\sessions\SESSION_ID"
$tc = Get-Content "$sess\analysis\teach_tool_calls.json" -Raw | ConvertFrom-Json
foreach ($c in $tc) {
    "[$($c.turn)] $($c.tool) status=$($c.status) dur=$($c.duration_sec)s"
}
```

Use this to debug "did the watch tool succeed?" or "what did the teacher actually look at?".

### Inspect comments

```powershell
$sess = "C:\Users\pqian\Source\music-masterclass-v2\local_adls\tenant\pqian\users\pqian\sessions\SESSION_ID"
$ce = Get-Content "$sess\lesson\comments_enriched.json" -Raw | ConvertFrom-Json
"summary: $($ce.summary)"
foreach ($c in $ce.comments) {
    $refs = ($c.references | ForEach-Object { "m$($_.measure)/p$($_.page)/s$($_.system_index)" }) -join '; '
    "  $($c.id) [$($c.severity)] $($c.summary) refs=[$refs]"
}
```

## Smoke tests

The `scripts/` directory has many `test_*.py` scripts that exercise individual stages.

| Script | What it tests |
|---|---|
| `test_audiveris_pipeline.py` | OMR + MusicXML → score_prep |
| `test_hmm_align.py` | HMM Viterbi |
| `test_intonation.py` | Cent deviation measurement |
| `test_rhythm.py` | Per-bar tempo + outlier detection |
| `test_voicing.py` | Chord-balance for piano |
| `test_score_map.py` | score_map.json generation |
| `test_mechanical_comments.py` | c001..c118 generation |
| `test_teach_agentic.py` | End-to-end teacher call |
| `test_listen_tool.py` | `listen` tool against a real session |
| `test_watch_tool.py` | `watch` tool against a real session |
| `test_midi_find.py` | Mutopia search + Gemini pick |
| `test_mutopia_parser.py` | Just the regex parser, no Gemini |
| `barline_detection_prototype.py` | 4-algorithm bake-off, writes visualizations |

All require the bundled Python: `tools\python\python.exe scripts\test_X.py`.

## Reading the architecture

The pipeline is best understood by reading these in order:

1. `apps/api/main.py:_run_lesson_jobs` (~250 lines) — the orchestrator
2. `engine/ingest.py` — media extraction
3. `engine/analysis.py` — initial analysis stages
4. `engine/hmm_align.py` — alignment (the deepest module)
5. `engine/score_prep.py` — score preparation cascade
6. `engine/teach_lesson.py` — teacher orchestration
7. `agent_tools/registry.py` + 11 tool files — what the teacher can call
8. `apps/api/static/player.html` — the consumer of all artifacts
