# Repair / re-run scripts

These are operational scripts for re-running parts of the lesson pipeline against an existing session, without re-extracting media or re-running upstream stages. Useful when:

- You changed teacher prompts and want to regenerate comments without re-aligning audio
- You fixed an alignment bug and want to re-run HMM + downstream
- You changed the score-map builder and want to rebuild without touching alignment

| Script | What it does |
|---|---|
| [`install_tools.ps1`](./install_tools.ps1) | Bootstrap the bundled toolchain (Python, ffmpeg, JRE, Audiveris). Run this first on a fresh clone. |
| [`rebuild_score_map.py`](./rebuild_score_map.py) | Rebuild `score/score_map.json` only. |
| [`rerun_hmm_align.py`](./rerun_hmm_align.py) | Re-run HMM alignment only. Pass `--no-refine` to skip onset refinement. |
| [`rerun_teacher.py`](./rerun_teacher.py) | Re-run intonation, rhythm, voicing, mechanical comments, score_map, prior-lessons rebuild, and the agentic teacher. The full post-HMM pipeline. |

## Usage

```powershell
# After completing a lesson, you find a bug. Fix the bug, then re-run:
tools\python\python.exe scripts\rerun_teacher.py <SESSION_ID> --tenant default --user default
```

Each script reads `MASTERCLASS_LOCAL_ADLS_ROOT` (default: `./local_adls`) for storage and `.env` for `GEMINI_API_KEY`.

## Why is this folder mostly empty?

The author's working tree contains ~30 additional one-off test scripts with hardcoded local paths (test recordings, IMSLP downloads, etc.). They are excluded from version control via `.gitignore`. Only the scripts that are useful to other users are tracked here.

If you want to add your own one-off scripts, just drop them in this folder — they will be ignored automatically. To track a new script, add an explicit `!scripts/your_script.py` exception in the top-level `.gitignore`.
