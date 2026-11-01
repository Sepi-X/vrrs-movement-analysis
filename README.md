# Markerless Movement Analysis for VR Rehabilitation (VRRS)

**Extracting movement data from ordinary videos of Virtual Reality Rehabilitation System (VRRS) sessions, using computer vision instead of extra sensors.**

Bachelor's thesis in Computer Science (Data Analysis curriculum), University of Messina, 2025/2026.
*"Towards Investigating the Role of Artificial Intelligence in Neurological Rehabilitation"*
Author: **Sepehr Mojdehdehbaher** · Supervisor: Prof. Antonio Celesti

📄 [Thesis (PDF)](docs/thesis.pdf) · 📊 [Presentation slides (PDF)](docs/presentation.pdf)

---

## The problem

VRRS platforms are used in neurological rehabilitation (stroke, Parkinson's disease, brain injury). Patients move a **keywand with a blue light sphere** to play therapeutic games. Therapists judge movement quality mostly by eye, and objective motion capture usually needs expensive marker-based systems.

This project asks: **can useful movement measures be extracted from the session video alone, with free tools and a normal webcam?**

## What the pipeline does

```
Session video (OBS, 30/60 fps)
   │
   ├─► Ball tracking (OpenCV)        HSV colour mask → morphology → contour/circle fit
   │                                  → Kalman filter → position, speed, path, depth factor
   │
   ├─► Spatial calibration           known sphere diameter (50 mm) → pixels-per-mm,
   │                                  robust rolling median per frame
   │
   ├─► Pose estimation (MediaPipe)   shoulder, elbow, wrist landmarks → joint angles
   │
   └─► Signal processing (SciPy)     interpolation to 20 Hz, FFT, band-pass 3.5–7.5 Hz,
                                      movement segmentation, quality metrics
            │
            ▼
   CSV (per frame) + JSON (session report)
```

About 2,200 lines of Python, organised in classes for calibration (`SpatialCalibrator`), tracking (`BlueSphereTracker`), pose analysis (`JointAngleAnalyzer`), tracking quality (`TrackingQualityAssessor`) and movement analytics.

## Results (from the thesis)

Data: **50 participants** (46 healthy volunteers, 4 people with Parkinson's disease), **51 recorded sessions**; **34 sessions** recorded with the same setup were used for the main analysis (30 volunteer + 4 patient sessions). All recordings use anonymous codes.

| Finding | Detail |
|---|---|
| **Speed** | Ball tracking **0.58 ms/frame** vs MediaPipe **17.9 ms/frame** (300-frame benchmark). Both run faster than 30 fps webcam video on a laptop. |
| **Tracking accuracy** | In 50 hand-marked frames, the tracker followed the sphere in **49/50**; stored positions were a median of **~21 mm** from the marked centre (raw colour detections ~14 mm; the Kalman filter lags fast movements). |
| **Calibration scope** | Converting pixels to mm matters for keywand displacement, speed and amplitude, but **not for joint angles**, which are invariant to uniform scaling. |
| **Tremor** | FFT-based tremor detection is **not suitable during active, goal-directed exercises**: voluntary movement and tracking artefacts also put energy in the 3.5–7.5 Hz band. Rest periods or sensors would be needed. |
| **Main limitation** | Colour segmentation depends on lighting (the bright centre of the sphere saturates and drops out of the mask). Detection rates ranged from 20% to ~100% per session. |

**Important:** with only 4 patients, this is a **technical feasibility study**. No clinical conclusions are drawn about differences between patients and volunteers.

## Repository

```
├── vrrs_tracking.py                    # main pipeline (run this)
├── notebooks/
│   ├── vrrs_notebook.ipynb             # step-by-step version with detailed result cells
│   └── performance_comparison.ipynb    # MediaPipe vs ball-tracking benchmark (Table 4.2)
├── docs/
│   ├── thesis.pdf
│   └── presentation.pdf
└── requirements.txt
```

Session videos are **not** included (participant privacy).

## Run it

Requires **Python 3.9–3.12** (MediaPipe 0.10.21, the last version with the `solutions` API used here, has no wheels for 3.13+).

```bash
git clone https://github.com/Sepi-X/vrrs-movement-analysis.git
cd vrrs-movement-analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python vrrs_tracking.py path/to/session.mkv --session-id 7VM
python vrrs_tracking.py --help          # all options (frame skip, scale, manual px/mm, headless...)
```

While running: `q` quit · `p` pause · `r` live report. Use `--no-display` on a server.

**Output:** `<session>_<exercise>_<timestamp>.csv` (per-frame positions, speed, joint angles) and `.json` (calibration, data quality, movement metrics).

The tracking region coordinates (`blue_roi`) match the screen layout of the VRRS recordings used in the thesis; adjust them in `track_blue_sphere_with_parkinson_analysis()` for a different layout.

## About the screening output

The JSON report also contains a heuristic "screening" classification (tremor / bradykinesia / dyskinesia). It is **experimental and not clinically validated**: it was not used for the conclusions of the thesis, and, as explained above, the tremor component is unreliable during active movement. It must not be used for diagnosis.

## Next steps

Clinical validation with more patients and paired MDS-UPDRS scores; rest periods in the exercise protocol for tremor; learned detectors (YOLO / U-Net) to replace colour thresholds; MediaPipe Hands for finger landmarks; monocular depth estimation.

## Tech

Python · OpenCV · MediaPipe · NumPy · SciPy · pandas · Jupyter
