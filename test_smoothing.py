"""
Fast, model-free check of the moving-average smoothing logic used in
yolo_demo_v2_smoothed.py / measure_system_metrics.py.

Verifies the core hypothesis mechanically (does averaging suppress a
single-frame spike while still responding to sustained crowding) before
spending time on real video measurement.

Run: python test_smoothing.py
"""

from collections import deque

SMOOTHING_WINDOW = 5
PERSON_THRESHOLD = 6


def moving_average(buffer):
    return sum(buffer) / len(buffer) if buffer else 0.0


def feed(sequence, window=SMOOTHING_WINDOW):
    """Feed a sequence of raw counts through the smoother, return list of smoothed values."""
    buf = deque(maxlen=window)
    out = []
    for v in sequence:
        buf.append(v)
        out.append(moving_average(buf))
    return out


def demo():
    # Case 1: empty buffer -> 0
    assert moving_average(deque()) == 0.0

    # Case 2: single-frame spike to 10 (well above threshold) surrounded by zeros
    # should NOT push the smoothed value over PERSON_THRESHOLD with window=5
    spike_seq = [0, 0, 0, 0, 10, 0, 0, 0, 0]
    smoothed = feed(spike_seq)
    peak = max(smoothed)
    assert peak < PERSON_THRESHOLD, (
        f"single-frame spike should be suppressed below threshold, got peak={peak}"
    )

    # Case 3: crowd jumps from 0 to 8 and stays there (step function).
    # Smoothed value should ramp up gradually and eventually cross threshold,
    # unlike raw counts which would cross it instantly on frame 1.
    step_seq = [0, 0, 0] + [8] * 10
    smoothed = feed(step_seq)
    assert smoothed[3] < PERSON_THRESHOLD, (
        f"first frame of the step should still be damped by prior zeros, got {smoothed[3]}"
    )
    assert smoothed[-1] >= PERSON_THRESHOLD, (
        f"sustained crowding should cross threshold once window fills, got {smoothed[-1]}"
    )
    assert smoothed[3] < smoothed[-1], "smoothed value should ramp up over the step, not jump instantly"

    # Case 4: raw counts are unaffected — moving_average is read-only over the buffer
    buf = deque([1, 2, 3], maxlen=5)
    _ = moving_average(buf)
    assert list(buf) == [1, 2, 3]

    print("All smoothing sanity checks passed.")
    print(f"  Spike test:     peak smoothed value = {peak:.2f} (threshold={PERSON_THRESHOLD})")
    print(f"  Sustained test: final smoothed value = {smoothed[-1]:.2f} (threshold={PERSON_THRESHOLD})")


if __name__ == "__main__":
    demo()
