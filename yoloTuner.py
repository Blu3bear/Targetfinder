"""
YOLO Model Tuner/Trainer for Target Detection.

This script provides a simple interface to either:
1. Run hyperparameter tuning using Ultralytics' built-in tune() method
2. Perform a single training run with specified hyperparameters

Usage:
    python yoloTuner.py --tune                    # Run hyperparameter tuning
    python yoloTuner.py --train                   # Train with default hyperparameters
    python yoloTuner.py --train --epochs 100      # Train for 100 epochs
"""

import argparse
import os

from ultralytics import YOLO

DEFAULT_MODEL = "yolo11n-obb.pt"  # YOLOv11 nano model, obb version


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the YOLO tuner/trainer.

    Args:
        args: Parsed command line arguments.
    """
    print("YOLO Target Detector Trainer")
    print(f"  Model: {args.model}")
    print()

    # Verify data.yaml exists
    assert os.path.isfile(
        "data.yaml"
    ), f"Error: data.yaml not found. Run targetGenerator.py first."

    # Load model
    model = YOLO(args.model)

    if args.tune:
        print(f"Starting hyperparameter tuning ({args.iterations} iterations)...")
        model.tune(
            data="data.yaml",
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            shear=0.1,
            translate=0.1,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            imgsz=args.img_size,
            device=0,
            batch=args.batch_size,
            iterations=args.iterations,
        )
    elif args.train:
        print(f"Starting training ({args.epochs} epochs)...")
        model.train(
            data="data.yaml",
            epochs=args.epochs,
            imgsz=args.img_size,
            device=0,
            batch=args.batch_size,
            workers = args.workers,
            patience = 30
        )
    elif args.validate:
        print("Running validation...")
        model.val("data.yaml",imgsz = args.img_size)
    else:
        print("No action specified. Use --tune, --train, or --validate.")
        print("Run with --help for more information.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or tune a YOLO model for target detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python yoloTuner.py --tune                     Run hyperparameter tuning
  python yoloTuner.py --tune --iterations 50     Run 50 tuning iterations
  python yoloTuner.py --train                    Train with default settings
  python yoloTuner.py --train --epochs 100       Train for 100 epochs
  python yoloTuner.py --validate                 Validate the current model
        """,
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--tune", help="Run hyperparameter tuning", action="store_true"
    )
    action_group.add_argument(
        "--train", help="Run a single training session", action="store_true"
    )
    action_group.add_argument(
        "--validate",
        help="Validate the model on the validation set",
        action="store_true",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        help=f"Model to use (default: {DEFAULT_MODEL})",
        default=DEFAULT_MODEL,
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        help=f"Number of training epochs (default: 50)",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch-size",
        help=f"Batch size for training (default: 14)",
        type=int,
        default=14,
        dest="batch_size",
    )
    parser.add_argument(
        "--img-size",
        help=f"Input image size (default: 640)",
        type=int,
        default=640,
        dest="img_size",
    )
    parser.add_argument(
        "--workers",
        help="Number of threads to train/tune on (default: 8)",
        type=int,
        default=8,
        dest="workers",
    )

    # Tuning parameters
    parser.add_argument(
        "--iterations",
        help="Number of tuning iterations (default: 30)",
        type=int,
        default=30,
    )

    args = parser.parse_args()
    main(args)
