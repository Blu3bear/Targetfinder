"""
Synthetic Data Generator for YOLO Target Detection Training.

This module generates synthetic training images by compositing procedural or
uploaded targets onto various backgrounds. It produces YOLO-format annotations
and manages the data directory structure required for training.

Usage:
    python targetGenerator.py -n 1000 -ts 80        # Generate 1000 images, 80% train / 20% val
    python targetGenerator.py -n 500 -c             # Clean existing data and generate 500 new images
    python targetGenerator.py -n 500                # Append 500 images to existing dataset
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

from PIL import Image
import numpy as np


# =============================================================================
# CONFIGURATION / CONSTANTS
# =============================================================================

DATA_DIR = Path("data")
OBJ_DIR = DATA_DIR / "obj"
TRAIN_TXT = DATA_DIR / "train.txt"
VALID_TXT = DATA_DIR / "valid.txt"
ALLDATA_TXT = DATA_DIR / "alldata.txt"
DATA_YAML = Path("data.yaml")

# Default image dimensions
DEFAULT_IMG_WIDTH = 640
DEFAULT_IMG_HEIGHT = 640

# Ukrainian flag colors (RGB)
UKRAINE_BLUE = (0, 87, 183)
UKRAINE_YELLOW = (255, 215, 0)

# Grass background base color (RGB)
GRASS_GREEN = (34, 139, 34)

# Augmentation probabilities (0.0 to 1.0)
AUGMENTATION_CONFIG = {
    "hue_shift": 0.3,
    "saturation_shift": 0.3,
    "noise_salt": 0.2,
    "random_lines": 0.2,
    "rotation": 0.4,
    "pixel_shift": 0.1,
}


# =============================================================================
# BACKGROUND GENERATION
# =============================================================================


def generate_grass_background(
    width: int = DEFAULT_IMG_WIDTH, height: int = DEFAULT_IMG_HEIGHT
) -> Image.Image:
    """
    Generate a procedural grass-like background image.

    Creates a green base layer with random noise overlay to simulate grass texture.

    Args:
        width: Width of the background image in pixels.
        height: Height of the background image in pixels.

    Returns:
        PIL Image of the generated grass background.
    """

    # Generate a noise map
    noise_map = np.random.normal(np.random.normal(0.75,0.1),40/255, (width,height))
    # Clip the noise map to be 0-1.0
    noise_map = np.clip(noise_map, 0, 1.0)
    # Expand map axis(1 channel image to 3 channel)
    noise_map = np.repeat(np.expand_dims(noise_map,axis=2),3,2)
    # Apply coloring
    colored_map = GRASS_GREEN*noise_map.astype(np.uint8)
    
    return Image.fromarray(colored_map)


def load_background_images(background_dir: Optional[Path] = None) -> List[Image.Image]:
    """
    Load background images from a directory, or generate procedural ones.

    Args:
        background_dir: Optional path to directory containing background images.
                       If None or empty, generates procedural backgrounds.

    Returns:
        List of PIL Images to use as backgrounds.
    """
    # TODO: Implement background loading
    # - Check if directory exists and has images
    # - Load all valid image files
    # - Fall back to procedural generation if needed
    raise NotImplementedError("load_background_images not yet implemented")


def get_random_background(backgrounds: List[Image.Image]) -> Image.Image:
    """
    Select a random background from the available options.

    Args:
        backgrounds: List of available background images.

    Returns:
        A copy of a randomly selected background image.
    """
    # TODO: Implement random background selection
    raise NotImplementedError("get_random_background not yet implemented")


# =============================================================================
# TARGET GENERATION
# =============================================================================


def generate_ukrainian_flag(width: int = 90, height: int = 60) -> Image.Image:
    """
    Generate a Ukrainian flag image (blue top half, yellow bottom half).

    Args:
        width: Width of the flag in pixels.
        height: Height of the flag in pixels.

    Returns:
        PIL Image of the Ukrainian flag with transparency.
    """

    top_half = np.ones((int(height/2),width,3))*UKRAINE_BLUE
    bottom_half = np.ones((int(height/2),width,3))*UKRAINE_YELLOW

    flag = np.concatenate((top_half,bottom_half))

    return Image.fromarray(flag.astype(np.uint8))


def load_target_images(target_dir: Optional[Path] = None) -> List[Image.Image]:
    """
    Load target images from a directory, or generate procedural ones.

    Args:
        target_dir: Optional path to directory containing target images.
                   If None or empty, generates procedural targets (Ukrainian flag).

    Returns:
        List of PIL Images to use as targets.
    """
    # TODO: Implement target loading
    # - Check if directory exists and has images
    # - Load all valid image files
    # - Fall back to procedural generation if needed
    raise NotImplementedError("load_target_images not yet implemented")


def get_random_target(targets: List[Image.Image]) -> Image.Image:
    """
    Select a random target from the available options.

    Args:
        targets: List of available target images.

    Returns:
        A copy of a randomly selected target image.
    """
    # TODO: Implement random target selection
    raise NotImplementedError("get_random_target not yet implemented")


# =============================================================================
# IMAGE COMPOSITION
# =============================================================================


def transform_target(
    target: Image.Image,
    scale_range: Tuple[float, float] = (0.1, 0.5),
    rotation_range: Tuple[float, float] = (0, 360),
) -> Image.Image:
    """
    Apply random scale and rotation transformations to the target.

    Args:
        target: The target image to transform.
        scale_range: Min and max scale factors (relative to original size).
        rotation_range: Min and max rotation angles in degrees.

    Returns:
        Transformed target image.
    """
    # TODO: Implement target transformation
    # - Random scaling within range
    # - Random rotation within range
    # - Maintain transparency
    raise NotImplementedError("transform_target not yet implemented")


def place_target_on_background(
    background: Image.Image, target: Image.Image
) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
    """
    Place the target at a random position on the background.

    Args:
        background: The background image.
        target: The target image to place (should have transparency).

    Returns:
        Tuple of (composite_image, yolo_bbox) where yolo_bbox is
        (x_center, y_center, width, height) normalized to [0, 1].
    """
    # TODO: Implement target placement
    # - Calculate valid placement region (target fully within background)
    # - Choose random position
    # - Composite target onto background
    # - Calculate and return YOLO format bounding box
    raise NotImplementedError("place_target_on_background not yet implemented")


# =============================================================================
# AUGMENTATIONS
# =============================================================================


def apply_augmentations(
    image: Image.Image, config: dict = AUGMENTATION_CONFIG
) -> Image.Image:
    """
    Apply random augmentations to the image based on configured probabilities.

    Args:
        image: The image to augment.
        config: Dictionary of augmentation names to probabilities.

    Returns:
        Augmented image.
    """
    # TODO: Implement augmentation pipeline
    # - For each augmentation, roll dice against probability
    # - Apply augmentation if selected
    raise NotImplementedError("apply_augmentations not yet implemented")


def augment_hue_shift(
    image: Image.Image, shift_range: Tuple[int, int] = (-30, 30)
) -> Image.Image:
    """
    Shift the hue of the image by a random amount.

    Args:
        image: Input image.
        shift_range: Min and max hue shift values.

    Returns:
        Hue-shifted image.
    """
    # TODO: Implement hue shift
    raise NotImplementedError("augment_hue_shift not yet implemented")


def augment_saturation(
    image: Image.Image, factor_range: Tuple[float, float] = (0.5, 1.5)
) -> Image.Image:
    """
    Adjust the saturation of the image by a random factor.

    Args:
        image: Input image.
        factor_range: Min and max saturation factors.

    Returns:
        Saturation-adjusted image.
    """
    # TODO: Implement saturation adjustment
    raise NotImplementedError("augment_saturation not yet implemented")


def augment_salt_noise(image: Image.Image, density: float = 0.01) -> Image.Image:
    """
    Add salt (white pixel) noise to the image.

    Args:
        image: Input image.
        density: Proportion of pixels to affect (0.0 to 1.0).

    Returns:
        Noisy image.
    """
    # TODO: Implement salt noise
    raise NotImplementedError("augment_salt_noise not yet implemented")


def augment_random_lines(
    image: Image.Image, num_lines_range: Tuple[int, int] = (1, 5)
) -> Image.Image:
    """
    Draw random lines on the image.

    Args:
        image: Input image.
        num_lines_range: Min and max number of lines to draw.

    Returns:
        Image with random lines.
    """
    # TODO: Implement random line drawing
    raise NotImplementedError("augment_random_lines not yet implemented")


def augment_rotation(
    image: Image.Image, angle_range: Tuple[float, float] = (-15, 15)
) -> Image.Image:
    """
    Rotate the entire image by a random angle.

    Args:
        image: Input image.
        angle_range: Min and max rotation angles in degrees.

    Returns:
        Rotated image (cropped to original dimensions).
    """
    # TODO: Implement image rotation
    raise NotImplementedError("augment_rotation not yet implemented")


def augment_pixel_shift(image: Image.Image, max_shift: int = 5) -> Image.Image:
    """
    Shift a layer of pixels by a random amount.

    Args:
        image: Input image.
        max_shift: Maximum number of pixels to shift.

    Returns:
        Pixel-shifted image.
    """
    # TODO: Implement pixel shift
    raise NotImplementedError("augment_pixel_shift not yet implemented")


# =============================================================================
# ANNOTATION & FILE MANAGEMENT
# =============================================================================


def generate_yolo_annotation(
    class_id: int, bbox: Tuple[float, float, float, float]
) -> str:
    """
    Generate a YOLO format annotation string.

    Args:
        class_id: The class ID for the object (0 for TARGET).
        bbox: Bounding box as (x_center, y_center, width, height) normalized to [0, 1].

    Returns:
        YOLO annotation string: "class_id x_center y_center width height"
    """
    # TODO: Implement YOLO annotation formatting
    raise NotImplementedError("generate_yolo_annotation not yet implemented")


def get_next_image_index(obj_dir: Path) -> int:
    """
    Find the next available image index when appending to existing dataset.

    Scans the obj directory for existing images and returns the next index.

    Args:
        obj_dir: Path to the data/obj directory.

    Returns:
        Next available integer index for image naming.
    """
    # TODO: Implement index scanning
    # - Find all existing img_*.jpg files
    # - Extract indices and find max
    # - Return max + 1 (or 0 if empty)
    raise NotImplementedError("get_next_image_index not yet implemented")


def save_image_and_annotation(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    index: int,
    obj_dir: Path = OBJ_DIR,
) -> Path:
    """
    Save an image and its corresponding YOLO annotation file.

    Args:
        image: The generated image to save.
        bbox: YOLO format bounding box.
        index: Image index for filename.
        obj_dir: Directory to save files to.

    Returns:
        Path to the saved image file.
    """
    # TODO: Implement saving
    # - Save image as img_{index}.jpg
    # - Save annotation as img_{index}.txt
    raise NotImplementedError("save_image_and_annotation not yet implemented")


def update_data_files(
    image_paths: List[Path], training_split: int, append: bool = True
) -> None:
    """
    Update train.txt, valid.txt, and alldata.txt with new image paths.

    Args:
        image_paths: List of paths to the generated images.
        training_split: Percentage of images to use for training (0-100).
        append: If True, append to existing files. If False, overwrite.
    """
    # TODO: Implement file updates
    # - Shuffle the image paths
    # - Split according to training_split percentage
    # - Write/append to train.txt and valid.txt
    # - Write/append all to alldata.txt
    raise NotImplementedError("update_data_files not yet implemented")


def update_data_yaml(num_classes: int = 1, class_names: List[str] = ["TARGET"]) -> None:
    """
    Update or create the data.yaml configuration file.

    Args:
        num_classes: Number of object classes.
        class_names: List of class names.
    """
    # TODO: Implement data.yaml update
    raise NotImplementedError("update_data_yaml not yet implemented")


# =============================================================================
# DATA DIRECTORY MANAGEMENT
# =============================================================================


def setup_data_directory(clean: bool = False) -> None:
    """
    Set up the data directory structure.

    Creates data/obj/ directory if it doesn't exist.
    If clean=True, deletes and recreates the directory.

    Args:
        clean: If True, remove existing data and start fresh.
    """
    # TODO: Implement directory setup
    # - Check if directories exist
    # - If clean, delete and recreate
    # - If not clean and doesn't exist, create
    raise NotImplementedError("setup_data_directory not yet implemented")


def clear_data_files() -> None:
    """
    Clear the contents of train.txt, valid.txt, and alldata.txt.
    """
    # TODO: Implement file clearing
    raise NotImplementedError("clear_data_files not yet implemented")


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================


def generate_single_image(
    backgrounds: List[Image.Image],
    targets: List[Image.Image],
    apply_augmentation: bool = True,
) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
    """
    Generate a single synthetic training image.

    Args:
        backgrounds: List of available background images.
        targets: List of available target images.
        apply_augmentation: Whether to apply random augmentations.

    Returns:
        Tuple of (generated_image, yolo_bbox).
    """
    # TODO: Implement single image generation pipeline
    # 1. Pick random background
    # 2. Pick random target
    # 3. Transform target (scale, rotate)
    # 4. Place target on background
    # 5. Apply augmentations if enabled
    raise NotImplementedError("generate_single_image not yet implemented")


def generate_dataset(num_images: int, training_split: int, clean: bool = False) -> None:
    """
    Generate a complete dataset of synthetic training images.

    Args:
        num_images: Total number of images to generate.
        training_split: Percentage of images for training (rest for validation).
        clean: If True, clear existing data before generating.
    """
    # TODO: Implement full dataset generation
    # 1. Setup/clean data directory
    # 2. Load or generate backgrounds and targets
    # 3. Determine starting index
    # 4. Generate each image
    # 5. Save images and annotations
    # 6. Update data files
    raise NotImplementedError("generate_dataset not yet implemented")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the targetGenerator.

    Args:
        args: Parsed command line arguments.
    """
    print(f"Target Generator")
    print(f"  Images to generate: {args.number}")
    print(f"  Training split: {args.training_split}%")
    print(f"  Clean mode: {args.clean}")
    print()

    generate_dataset(
        num_images=args.number, training_split=args.training_split, clean=args.clean
    )

    print("Dataset generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for YOLO target detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python targetGenerator.py -n 1000 -ts 80     Generate 1000 images (80%% train, 20%% val)
  python targetGenerator.py -n 500 -c          Clean existing data, generate 500 new images
  python targetGenerator.py -n 500             Append 500 images to existing dataset
        """,
    )
    parser.add_argument(
        "-c",
        "--clean",
        help="Clean data directory before generating (removes existing images)",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--number",
        help="Total number of images to generate (default: 100)",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-ts",
        "--training_split",
        help="Percentage of images for training, rest for validation (default: 80)",
        default=80,
        type=int,
        choices=range(0, 101),
        metavar="0-100",
    )

    args = parser.parse_args()
    main(args)
