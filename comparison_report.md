# Phase 2 Comparison Report: Baseline vs Temporal Smoothing

## Methodology

- Same video processed with identical YOLO model, confidence (0.5), grid (4x4), thresholds.
- Baseline: raw per-frame detection count drives alert decisions (yolo_demo.py logic).
- V2: 5-frame moving average of detection count drives alert decisions (yolo_demo_v2_smoothed.py logic).
- Frames compared: 392

**Note on false positives:** this is a real (non-synthetic) crowd video without per-frame ground truth annotations, so we do NOT claim a false-positive rate. Instead we report transient alert events (runs of <=2 frames), alert event counts, alert duration, and detection stability, per the measurement methodology.

## Global Crowd Alert

| Metric | Baseline | V2 (Smoothed) |
|---|---|---|
| total_alert_events | 6 | 3 |
| transient_events_le2frames | 1 | 0 |
| transient_fraction_of_events | 0.167 | 0.0 |
| mean_alert_duration_frames | 64 | 128.67 |
| median_alert_duration_frames | 30.5 | 119 |
| max_alert_duration_frames | 222 | 221 |
| mean_alert_duration_sec | 2.135 | 4.293 |
| pct_frames_in_alert | 97.96 | 98.47 |

## Zone Overcrowd Alert

| Metric | Baseline | V2 (Smoothed) |
|---|---|---|
| total_alert_events | 19 | 9 |
| transient_events_le2frames | 7 | 1 |
| transient_fraction_of_events | 0.368 | 0.111 |
| mean_alert_duration_frames | 17.16 | 31.22 |
| median_alert_duration_frames | 4 | 8 |
| max_alert_duration_frames | 130 | 127 |
| mean_alert_duration_sec | 0.573 | 1.042 |
| pct_frames_in_alert | 83.16 | 71.68 |

## Detection Variability (context — identical detections feed both systems)

| Metric | Baseline | V2 (Smoothed) |
|---|---|---|
| mean_abs_frame_to_frame_delta | 0.76 | 0.76 |
| stdev_person_count | 1.767 | 1.767 |
| max_person_count | 14 | 14 |
| min_person_count | 5 | 5 |

## Latency / Throughput

| Metric | Baseline | V2 (Smoothed) |
|---|---|---|
| mean_inference_ms | 24.931 | 24.537 |
| median_inference_ms | 22.645 | 22.569 |
| max_inference_ms | 912.229 | 768.947 |

## Interpretation

**What changed:** Only the alert-decision layer — a 5-frame moving average of
detection counts replaces the raw per-frame count when checking
`PERSON_THRESHOLD` and `ZONE_THRESHOLD`. Detection (YOLO model, confidence,
grid, box drawing) is byte-identical between the two runs.

**Test video characteristics (matters for reading these numbers):** `crowd.mp4`
is a persistently crowded scene — raw person count ranges 5–14 across all 392
frames, i.e. it is at or above the global threshold (6) almost the entire
video. This is NOT a sparse scene with occasional crowding; it's closer to a
stress test for a system that's basically always "on." That limits what this
run can say about sparse/intermittent scenarios — see limitations below.

**Zone alert — the clearest, most interpretable result:**
- Transient events (≤2 frames) dropped from 7/19 (36.8%) to 1/9 (11.1%) of all
  zone-alert events.
- Total zone-alert events roughly halved: 19 → 9.
- Time spent in zone alert actually *decreased*: 83.2% → 71.7% of frames.
- Median alert duration doubled (4 → 8 frames) — fewer, more sustained alerts
  instead of many short ones.

This is consistent with the hypothesis: smoothing suppressed short spikes in
per-zone occupancy without simply prolonging every alert indefinitely — total
alert-time went down, not up.

**Global alert — smaller/different effect:**
- Total events roughly halved (6 → 3) and the one transient event disappeared,
  but % of frames in alert barely moved (97.96% → 98.47%, technically higher).
  Because the raw count sits above threshold almost the whole video, smoothing
  mostly merged brief below-threshold dips into continuous alert stretches
  rather than removing alerts outright. **This metric alone would look like
  smoothing "did nothing" or made it slightly worse** if event count wasn't
  reported alongside it — worth remembering when interpreting single metrics.

**Detection variability:** Identical between runs (0.76 mean frame-to-frame
delta, stdev 1.767, same min/max), confirming the comparison isolates the
smoothing effect cleanly rather than measuring detection noise.

**Latency:** 24.9ms → 24.5ms mean inference — no measurable cost from
smoothing (within run-to-run noise; moving-average over a 5-item deque is
negligible next to ~25ms YOLO inference). *(An earlier run showed a large gap
here because the baseline run's model weights were still downloading
mid-measurement — re-run after caching the model to get this clean number;
noted so the methodology stays transparent about the mistake.)*

### Limitations

- **Single video, single run each.** No repeated trials, no confidence
  intervals. The percentages above are one data point, not a robust estimate.
- **No independent ground truth.** We cannot and do not claim a "false
  positive rate" — only alert event counts, durations, and transient fraction,
  as scoped in the experiment methodology.
- **Small N for zone-alert events** (19 vs 9 total events, 7 vs 1 transient).
  Percentage swings on small counts are more sensitive to individual
  borderline frames than they'd be with more events.
- **One scene type.** `crowd.mp4` is persistently near/above threshold. A
  sparser video (crowd forms and disperses repeatedly) would be a more direct
  test of "does smoothing kill single-frame false alarms" and might show a
  larger or smaller effect than measured here.

### Recommendation for Tier 2 (frame persistence / hysteresis)

Zone-alert transient fraction dropped substantially (36.8% → 11.1%) from
smoothing alone, but did not reach zero — one transient event remains, and
median zone-alert duration (8 frames) is still short. **Tier 2 is plausibly
worthwhile for zone alerts** if further reducing that remaining transient
event matters for the use case, since persistence/hysteresis targets exactly
this residual flicker.

For the global alert, this video doesn't exercise the persistence question
meaningfully — the scene is almost always crowded, so there's little
"dip-then-recover" pattern for hysteresis to act on. Testing Tier 2 would need
a video with more genuine crowd formation/dispersal cycles to be informative.

**Suggested next step before building Tier 2:** run this same comparison on a
second, sparser video (or a longer clip with clearer crowd/no-crowd cycles) to
see if the zone-alert improvement replicates, rather than committing to Tier 2
on the strength of one run.
