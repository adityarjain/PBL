# Claude Code Task: Crowd Monitoring PBL Project - Phase 1 Completion

## PROJECT CONTEXT

**Project:** AI-Based Crowd Monitoring and Safety Alert System  
**Student:** Aditya Raj Jain (B.Tech CSE, Manipal University Jaipur)  
**Supervisor:** Dr. Mohit Kushwaha  
**Status:** Post-presentation, needs stabilization and evaluation  

## TASK OVERVIEW

Complete Phase 1 of the project takeover: stabilize the codebase, fix critical bug, set up proper project structure, and prepare for baseline evaluation.

---

## PHASE 1: STABILIZATION & SETUP (Current Status)

### COMPLETED ✓
- [x] Fixed critical color assignment bug in `yolo_demo.py` (lines 68-82)
  - Issue: `elif count == 1: continue` caused undefined `color` variable
  - Fix: Restructured to `if count == 0: continue`, then always assign color
  - Verified: Code now handles all cases (0, 1, 2, ≥3 people per zone)
  
- [x] Created `requirements.txt` with dependencies
  - ultralytics>=8.0.0
  - opencv-python>=4.8.0
  - numpy

- [x] Created `.gitignore` file
  - Excludes: venv/, __pycache__/, .pyc, yolov8n.pt, model cache
  
- [x] Updated `README.md` with proper documentation
  - Installation instructions
  - Quick start guide
  - System overview
  - Current status & next steps

- [x] Git repository initialized
  - User configured (Aditya Raj Jain)
  - Files staged

### TODO - NEXT STEPS IN CLAUDE CODE

1. **Verify & Test** (10 min)
   - [ ] Run syntax check on fixed `yolo_demo.py`
   - [ ] Verify all files are present and correct
   - [ ] Commit changes with message: "Phase 1: Fix grid overlay bug + project setup"
   - [ ] Verify git log shows commit

2. **Create Baseline Evaluation Script** (30 min)
   - [ ] Create `evaluate_baseline.py`
   - [ ] Measure:
     - Average FPS
     - Inference time per frame
     - Total people detected (from webcam or test video)
     - Memory usage (optional)
   - [ ] Output results to console + save to `baseline_results.json`
   - [ ] Document exact conditions (resolution, model, confidence threshold)

3. **Test Evaluation Script** (15 min)
   - [ ] Run on sample video or webcam
   - [ ] Verify metrics are recorded correctly
   - [ ] Document actual results (NO claimed values, only measured)

4. **Create Experimental Framework** (20 min)
   - [ ] Create `experiment_confidence_thresholds.py`
   - [ ] Test confidence thresholds: 0.3, 0.5, 0.7
   - [ ] Measure detection count, FPS impact for each
   - [ ] Save results to `conf_threshold_results.csv`

5. **Documentation Update** (10 min)
   - [ ] Update README with evaluation instructions
   - [ ] Add section "Measured Performance"
   - [ ] Document how to run experiments

6. **Final Commit** (5 min)
   - [ ] Commit all evaluation code
   - [ ] Message: "Phase 1 Complete: Baseline evaluation framework ready"

---

## DETAILED SPECIFICATIONS

### 1. SYNTAX CHECK & GIT COMMIT

**File to verify:** `/home/claude/yolo_demo.py`

**What to check:**
- No syntax errors
- Bug fix properly applied (lines 70-89)
- All imports present
- Grid overlay logic complete

**Git commands:**
```bash
cd /home/claude
git status
git commit -m "Phase 1: Fix grid overlay bug + project setup

- Fixed color assignment bug in grid overlay (count==1 case)
- Restructured logic to always assign color before drawing
- Added requirements.txt with dependencies
- Created .gitignore (excludes venv, cache, model files)
- Updated README.md with installation and usage instructions
- Initialized git repository with proper user config

This commit stabilizes the codebase for baseline evaluation."

git log --oneline -5
```

---

### 2. BASELINE EVALUATION SCRIPT

**File to create:** `evaluate_baseline.py`

**Purpose:** Establish measurable baseline before any improvements

**What to measure:**
```python
metrics = {
    "model": "yolov8n",
    "confidence_threshold": 0.5,
    "input_resolution": "detected_automatically",
    "total_frames": 0,
    "total_time_seconds": 0.0,
    "average_fps": 0.0,
    "average_inference_ms": 0.0,
    "total_people_detected": 0,
    "average_people_per_frame": 0.0,
    "memory_usage_mb": 0.0,  # optional
    "test_date": "2026-09-08",
    "test_source": "webcam or video file",
}
```

**Script outline:**
```python
1. Load YOLO model
2. Open video source (webcam or test video)
3. Loop through frames:
   - Measure inference time
   - Count detected people
   - Track FPS
   - Monitor memory (optional)
4. Calculate averages after loop
5. Save to JSON
6. Print summary to console
```

**Key requirement:** Measure actual values, NO assumptions or claims.

---

### 3. TEST EVALUATION

**What to do:**
- [ ] Run `evaluate_baseline.py` with webcam for 30-60 seconds
- [ ] Record the output metrics
- [ ] Document exact conditions (webcam resolution, lighting, crowd size)
- [ ] Save baseline results

**Expected output format:**
```
=== YOLO Baseline Evaluation ===
Model: yolov8n
Confidence Threshold: 0.5
Duration: 45.3 seconds
Total Frames: 1356
Average FPS: 30.0
Average Inference: 32.5ms
Total People Detected: 127
Average People/Frame: 2.8
Memory Usage: 850MB
```

---

### 4. CONFIDENCE THRESHOLD EXPERIMENT

**File to create:** `experiment_confidence_thresholds.py`

**Purpose:** Find optimal confidence threshold for this scenario

**Test each threshold:**
- 0.3 (low - more detections, more false positives)
- 0.5 (current - balanced)
- 0.7 (high - fewer false positives, might miss people)

**Measure for each:**
- Detection count
- FPS impact
- Whether zones trigger alerts differently

**Output:** CSV file
```csv
confidence_threshold,total_detections,avg_fps,avg_inference_ms,zones_alert_count
0.3,150,28.5,35.0,12
0.5,127,30.0,32.5,8
0.7,95,31.2,30.0,4
```

---

### 5. README UPDATE

**Add sections:**

```markdown
## Evaluation

### Baseline Performance

Run baseline evaluation:
```bash
python evaluate_baseline.py
```

This measures:
- FPS on your hardware
- Inference time per frame
- Average detection count
- Memory usage

Results saved to: `baseline_results.json`

### Confidence Threshold Optimization

Test different confidence thresholds:
```bash
python experiment_confidence_thresholds.py
```

Results saved to: `conf_threshold_results.csv`

This helps find the optimal balance between:
- Detection accuracy (catch more people)
- False positives (avoid noise)
- Performance (maintain FPS)

### Measured Performance (Current Hardware)

[Update after running evaluation]
- Average FPS: [measured]
- Inference Time: [measured]ms
- People Detected: [measured] avg per frame
- Date Measured: [date]
```

---

## IMPORTANT GUIDELINES

### DO:
✓ Measure actual values only
✓ Document exact test conditions
✓ Save results to files for record-keeping
✓ Test with real video/webcam
✓ Report what you actually find (good or bad)
✓ Use proper error handling

### DON'T:
✗ Claim performance you haven't measured
✗ Assume hardware performance
✗ Test on tiny/artificial video
✗ Fabricate numbers
✗ Skip edge cases
✗ Assume detection is perfect

---

## FILES AFTER PHASE 1

```
/home/claude/
├── yolo_demo.py              # ✓ Bug fixed
├── sample.py                 # HOG baseline (unchanged)
├── requirements.txt          # ✓ Created
├── .gitignore               # ✓ Created
├── README.md                # ✓ Updated
├── evaluate_baseline.py     # ← TODO
├── experiment_confidence_thresholds.py  # ← TODO
├── baseline_results.json    # ← Auto-generated
├── conf_threshold_results.csv  # ← Auto-generated
└── .git/                    # ✓ Repository
```

---

## SUCCESS CRITERIA

Phase 1 complete when:

- [x] Bug fixed and committed
- [x] Project structure clean
- [ ] Baseline evaluation script runs successfully
- [ ] Actual FPS, inference time, detection counts measured
- [ ] Results saved to JSON/CSV
- [ ] README updated with measured performance
- [ ] All changes committed with clear messages
- [ ] Ready to move to Phase 2: Algorithm Improvement

---

## PHASE 2 PREVIEW (After Phase 1)

Once Phase 1 is done, Phase 2 will involve:
1. Temporal smoothing (reduce frame-to-frame noise)
2. Tracking (maintain person identity across frames)
3. Zone persistence (require condition to persist N frames)
4. Performance comparison (measure improvement vs baseline)

But first, complete Phase 1.

---

## WORKING DIRECTORY

All work happens in: `/home/claude/`

Files are synced to: `/mnt/user-data/outputs/` for backup

Git repository: `/home/claude/.git/`

---

## ESTIMATED TIME

- Syntax check & commit: 5 min
- Baseline evaluation script: 30 min
- Testing & documentation: 20 min
- Confidence threshold experiment: 20 min
- Final cleanup: 10 min

**Total: ~85 minutes**

After completion, project moves from "buggy prototype" to "measurable research platform."
