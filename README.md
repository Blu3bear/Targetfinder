# Targetfinder

A synthetic data generator and YOLO trainer for target detection. This tool generates training images by compositing target objects onto various backgrounds, producing YOLO-format annotations for oriented bounding box (OBB) detection.

## Features

- **Synthetic Data Generation**: Automatically generate training images with randomized:
  - Target placement, rotation, and scaling
  - Procedural grass/gravel backgrounds or custom background images
  - Multiple target classes from directory structure

- **YOLO OBB Training**: Train YOLO models with oriented bounding boxes for rotated object detection

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Generate Training Data

```bash
# Generate 1000 images with 80% train / 20% validation split
python targetGenerator.py -n 1000 -ts 80

# Clean existing data and generate 500 new images
python targetGenerator.py -n 500 -c

# Append 500 images to existing dataset
python targetGenerator.py -n 500

# Verbose mode with custom directories
python targetGenerator.py -n 1000 -v --target-dir my_targets --bg-dir my_backgrounds
```

### Train YOLO Model

```bash
# Train with default settings (50 epochs)
python yoloTuner.py --train

# Train for 100 epochs
python yoloTuner.py --train --epochs 100

# Run hyperparameter tuning
python yoloTuner.py --tune --iterations 50

# Validate the model
python yoloTuner.py --validate
```

## Directory Structure

```
Targetfinder/
├── target/           # Place target images here (organized by class in subfolders)
├── backgrounds/      # Place background images here
├── data/             # Generated training data (auto-created)
│   ├── obj/          # Images and annotations
│   ├── train.txt     # Training image list
│   └── valid.txt     # Validation image list
├── data.yaml         # YOLO dataset configuration
├── targetGenerator.py
├── yoloTuner.py
└── requirements.txt
```

## Adding Custom Targets

1. Create subfolders in `target/` for each class
2. Add target images (PNG with transparency recommended)
3. Run `targetGenerator.py` to generate training data
