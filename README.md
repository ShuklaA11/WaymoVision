# Waymo Vehicle Detection Pipeline

*Developed as part of my research at INSPIRE Lab, UT Austin.*

A two-stage computer vision pipeline that detects Waymo self-driving cars (white Jaguar I-PACE with rooftop LIDAR dome) in street surveillance video and exports short clips of each pass.

The pipeline processes footage from an elevated intersection camera, identifies Waymo vehicles among regular traffic, and outputs organized video clips sorted by time of day (day/night).

**Core principle: precision over recall.** False positives (clipping a random white car) are far worse than missed detections. Every threshold is tuned to minimize false positives.

## How It Works

The pipeline uses two stages of classification to achieve high precision:

1. **Vehicle Detection + Tracking** -- YOLO26s detects all vehicles (cars, buses, trucks) and pedestrians. ByteTrack assigns persistent track IDs across frames so each vehicle is followed through the scene.

2. **Waymo Classification** -- A ResNet18 binary classifier examines each vehicle crop, looking for the distinctive rooftop LIDAR dome that distinguishes Waymo vehicles from regular cars. Crops are extended upward to capture the dome.

3. **Event Detection** -- Tracks that pass the Waymo classification threshold are converted into enter/exit events. Short tracks, low-confidence detections, and duplicate tracks are filtered out.

4. **Clip Extraction** -- ffmpeg extracts clips from the source video at 1080p/30fps with configurable padding before and after each event. Clips are sorted into `day/` and `night/` folders based on frame luminance.

### Additional Processing

- **Track interpolation** fills gaps where detection dropped for a few frames, producing smoother bounding boxes.
- **Cross-class NMS** removes duplicate boxes when the same vehicle is detected as both "car" and "truck."
- **Track deduplication** merges cases where ByteTrack assigns two IDs to the same vehicle.
- **Day/night classification** samples frame luminance across the video to determine time of day.

## Tech Stack

| Component | Technology |
|---|---|
| Object detection | [YOLO26](https://github.com/ultralytics/ultralytics) (via Ultralytics) |
| Object tracking | [ByteTrack](https://github.com/ifzhang/ByteTrack) (built into Ultralytics) |
| Waymo classifier | ResNet18 (PyTorch / torchvision) |
| Video I/O | OpenCV |
| Clip extraction | ffmpeg (external binary) |
| Configuration | Pydantic + YAML |
| GPU acceleration | PyTorch with CUDA / MPS / CPU auto-detection |

## Project Structure

```
waymo_detection/
├── config.yaml                ← all tunable parameters
├── requirements.txt           ← Python dependencies
├── waymo_pipeline/            ← main pipeline code
│   ├── run.py                 ← CLI entry point
│   ├── pipeline.py            ← orchestrator
│   ├── config.py              ← Pydantic config + YAML loading
│   ├── detect_and_track.py    ← YOLO26 + ByteTrack detection
│   ├── waymo_classifier.py    ← ResNet18 binary classifier
│   ├── event_detector.py      ← track-to-event conversion
│   ├── clip_extractor.py      ← ffmpeg clip cutting
│   ├── day_night.py           ← luminance-based day/night classification
│   └── track_interpolator.py  ← bbox gap interpolation
├── training/                  ← classifier training tools
│   ├── train_classifier.py    ← train ResNet18 waymo/not_waymo
│   ├── extract_crops.py       ← extract vehicle crops for labeling
│   ├── extract_white_crops.py ← extract only white vehicle crops
│   ├── filter_white_crops.py  ← filter crops by color
│   └── label_batches.py       ← batch labeling helper
├── models/                    ← trained model weights (not tracked in git)
├── data/                      ← raw input videos (not tracked in git)
├── output/                    ← pipeline output (not tracked in git)
│   ├── day/                   ← daytime Waymo clips
│   ├── night/                 ← nighttime Waymo clips
│   └── detection_log.csv      ← structured detection log
└── training/dataset/          ← labeled training images (not tracked in git)
    ├── waymo/
    └── not_waymo/
```

## Installation

### Prerequisites

- Python 3.10+
- ffmpeg (system dependency, not a Python package)

### 1. Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via `winget install ffmpeg` / `choco install ffmpeg`
2. Add the `bin/` folder to your system PATH
3. Verify: `ffmpeg -version`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

PyTorch will automatically install with CPU support. For GPU acceleration:

**NVIDIA GPU (CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon (MPS):**
```bash
# MPS support is included in the default PyTorch macOS build -- no extra steps needed.
pip install torch torchvision
```

### 3. Download or train model weights

The pipeline requires two model files in the project root and `models/` directory:

- A YOLO model (e.g., `yolo26s.pt`) -- auto-downloaded by Ultralytics on first run
- `models/waymo_classifier.pth` -- trained ResNet18 classifier (see [Training the Classifier](#training-the-classifier))

## Usage

### Run the full pipeline

```bash
# Process all videos in data/
python -m waymo_pipeline.run

# Process a single video
python -m waymo_pipeline.run --video data/my_video.mp4

# Use a custom config file
python -m waymo_pipeline.run --config custom_config.yaml

# Limit to first 2000 frames (for testing)
python -m waymo_pipeline.run --max-frames 2000

# Override compute device
python -m waymo_pipeline.run --device cpu
```

### Device selection

The pipeline auto-detects the best available compute device (CUDA > MPS > CPU). You can override this in `config.yaml`:

```yaml
device: "auto"    # auto-detect (default)
device: "cuda"    # force NVIDIA GPU
device: "mps"     # force Apple Silicon GPU
device: "cpu"     # force CPU
```

Or via CLI: `--device cuda`

### Training the Classifier

1. **Extract vehicle crops from your videos:**
```bash
python training/extract_crops.py --video data/my_video.mp4 --output training/dataset/unsorted/
```

2. **Sort crops into `waymo/` and `not_waymo/` folders** inside `training/dataset/`. The LIDAR dome on the roof is the key feature.

3. **Train:**
```bash
python training/train_classifier.py --data training/dataset/ --output models/waymo_classifier.pth
```

The trainer auto-detects your compute device. Training uses ImageNet-pretrained ResNet18 with heavy augmentation (color jitter, blur, random erasing) to handle noisy night footage.

## Configuration

All thresholds live in `config.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `classifier_confidence` | `0.75` | Minimum score to classify a crop as Waymo |
| `waymo_frame_ratio` | `0.4` | Fraction of track frames that must be Waymo-positive |
| `min_track_frames` | `8` | Minimum track length to consider |
| `max_clip_duration_sec` | `60` | Hard cap on output clip length |
| `gap_tolerance_frames` | `50` | Bridge tracking gaps up to ~2 seconds |
| `detection_confidence` | `0.35` | YOLO detection threshold (kept moderate; classifier does real filtering) |
| `day_night_luminance_threshold` | `80.0` | Mean grayscale below this = night |

## Output

The pipeline produces:

- **Video clips** in `output/day/` and `output/night/`, re-encoded to 1080p H.264 at 30fps
- **`output/detection_log.csv`** with per-clip metadata: track ID, frame range, confidence scores, waymo ratio

## License

This project is for research and educational purposes.
