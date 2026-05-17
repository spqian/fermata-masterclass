from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "engine" / "fixture_bach_matcher.json"
sys.path.insert(0, str(ROOT / "src"))

from masterclass.engine.audio_truth import match_to_score  # noqa: E402


def _legacy_greedy(perf_notes: list[dict], score_notes: list[dict]) -> list[dict]:
    by_pitch: dict[int, list[int]] = {}
    for idx, sn in enumerate(score_notes):
        by_pitch.setdefault(sn["midi_pitch"], []).append(idx)
    claimed = [False] * len(score_notes)
    lag = None
    last_score_t = -1e9
    out = []
    for pn in perf_notes:
        enriched = dict(pn)
        pitches = pn.get("pitches_midi") or []
        best_idx = None
        best_cost = float("inf")
        if pitches:
            pitch = int(pitches[0])
            perf_t = float(pn.get("performed_time_sec", pn.get("perf_time", 0.0)))
            for cand_idx in by_pitch.get(pitch, []):
                if claimed[cand_idx]:
                    continue
                sn = score_notes[cand_idx]
                score_t = float(sn["score_time_sec"])
                cost = abs(perf_t - (score_t + (lag or 0.0)))
                if cost > 30.0:
                    continue
                if last_score_t - score_t > 0.5:
                    cost += 6.0
                if cost < best_cost:
                    best_cost = cost
                    best_idx = cand_idx
        if best_idx is None:
            enriched["matched"] = False
            enriched["staff_index"] = None
        else:
            sn = score_notes[best_idx]
            claimed[best_idx] = True
            perf_t = float(pn.get("performed_time_sec", pn.get("perf_time", 0.0)))
            score_t = float(sn["score_time_sec"])
            lag = perf_t - score_t if lag is None else 0.85 * lag + 0.15 * (perf_t - score_t)
            last_score_t = max(last_score_t, score_t)
            enriched.update(
                matched=True,
                staff_index=sn.get("staff_index"),
                track_name=sn.get("track_name"),
                measure=sn.get("measure"),
                score_time_sec=score_t,
                score_midi_pitch=sn.get("midi_pitch"),
                timing_offset_ms=round((perf_t - score_t) * 1000.0, 1),
            )
        out.append(enriched)
    return out


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _spearman(rows: list[dict]) -> float:
    xs = [float(r["performed_time_sec"]) for r in rows]
    ys = [float(r["score_time_sec"]) for r in rows]
    rx = _rank(xs)
    ry = _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = (sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry)) ** 0.5
    return num / den


def _measure_table(rows: list[dict]) -> list[tuple[int, float, float, int]]:
    by_measure: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("matched") and row.get("measure") is not None:
            by_measure[int(row["measure"])].append(float(row["performed_time_sec"]))
    return [(m, min(ts), max(ts), len(ts)) for m, ts in sorted(by_measure.items())]


def _print_summary(label: str, rows: list[dict]) -> None:
    matched = [r for r in rows if r.get("matched")]
    print(f"{label}: {len(matched)}/{len(rows)} matched ({len(matched) / max(1, len(rows)):.1%})")
    print(f"{label}: Spearman(perf_time, score_time) = {_spearman(matched):.4f}")
    print("measure | first_perf_sec | last_perf_sec | matched")
    for measure, first, last, count in _measure_table(matched):
        print(f"{measure:>7} | {first:>14.3f} | {last:>13.3f} | {count:>7}")
    print()


def main() -> None:
    data = json.loads(FIXTURE.read_text())
    perf_notes = data["perf_notes"]
    score_notes = data["score_notes"]
    _print_summary("legacy greedy", _legacy_greedy(perf_notes, score_notes))
    _print_summary("banded NW", match_to_score(perf_notes, score_notes))


if __name__ == "__main__":
    main()
