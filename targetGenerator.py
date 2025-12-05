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

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import yaml
from PIL import Image

# =============================================================================
# CONFIGURATION / CONSTANTS
# =============================================================================

DATA_DIR = Path("data")
OBJ_DIR = DATA_DIR / "obj"
TRAIN_TXT = DATA_DIR / "train.txt"
VALID_TXT = DATA_DIR / "valid.txt"
ALLDATA_TXT = DATA_DIR / "alldata.txt"
DATA_YAML = Path("data.yaml")
TARGET_DIR = Path("target")
BG_DIR = Path("backgrounds")

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

VERBOSE = False

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
    noise_map = np.random.normal(np.random.normal(0.75, 0.1), 40 / 255, (width, height))
    # Clip the noise map to be 0-1.0
    noise_map = np.clip(noise_map, 0, 1.0)
    # Expand map axis(1 channel image to 3 channel)
    noise_map = np.repeat(np.expand_dims(noise_map, axis=2), 3, 2)
    # Apply coloring
    colored_map = GRASS_GREEN * noise_map

    return Image.fromarray(colored_map.astype(np.uint8))


def load_background_images(
    background_dir: Optional[Path] = None, num_bg: int = 5
) -> list[Image.Image]:
    """
    Load background images from a directory, or generate procedural ones.

    Args:
        background_dir: Optional path to directory containing background images.
                       If None or empty, generates procedural backgrounds.
        num_bg: The number of backgrounds to generate when generating backgrounds

    Returns:
        List of PIL Images to use as backgrounds.
    """

    background_list = []

    if VERBOSE:
        print("Loading in backgrounds...")
    # Check background directory
    if not background_dir or os.path.isdir(background_dir):
        # Directory exists, now check for not empty
        for file in os.scandir(background_dir):
            try:
                background_list.append(Image.open(file.path))
            except Exception as e:
                print(f"Couldn't open image {file.path} with exception {e}")
        if not background_list:
            print(
                f"WARNING: Background directory empty!!! \n Generating {num_bg} background images"
            )
            for _ in range(num_bg):
                background_list.append(generate_grass_background())
        elif VERBOSE:
            print(f"Done, found {len(background_list)} backgrounds!")
    else:
        # No directory generating images
        if VERBOSE:
            print(f"Generating {num_bg} background images")
        for _ in range(num_bg):
            background_list.append(generate_grass_background())

    return background_list


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

    top_half = np.ones((int(height // 2), width, 4)) * (*UKRAINE_BLUE, 255)
    bottom_half = np.ones((int(height // 2 + height % 2), width, 4)) * (
        *UKRAINE_YELLOW,
        255,
    )

    flag = np.concatenate((top_half, bottom_half))

    return Image.fromarray(flag.astype(np.uint8), mode="RGBA")


def load_target_images(
    target_dir: Optional[Path] = None,
) -> dict[str, list[Image.Image]]:
    """
    Load target images from a directory, or generate procedural ones.

    Args:
        target_dir: Optional path to directory containing target images.
                   If None or empty, generates procedural targets (Ukrainian flag).

    Returns:
        A dict containing classes mapping to lists of images to use for targets
    """

    target_dict = {"target": []} if not target_dir else {target_dir.name: []}

    if VERBOSE:
        print("Loading in targets...")
    # Check target directory
    if not target_dir or os.path.isdir(target_dir):
        # Directory exists, now check for not empty
        for file in os.scandir(target_dir):
            if not os.path.isdir(file):
                try:
                    target_dict[target_dir.name].append(Image.open(file.path))
                except Exception as e:
                    print(f"Couldn't open image {file.path} with exception {e}")
            else:
                target_dict[file.name] = []
                for target in os.scandir(file):
                    try:
                        target_dict[file.name].append(Image.open(target.path))
                    except Exception as e:
                        print(f"Couldn't open image {file.path} with exception {e}")
        if not any(target_dict.values()):
            print(
                f"WARNING: Target directory empty!!! \n Generating a Ukraine Flag image"
            )
            target_dict[target_dir.name].append(generate_ukrainian_flag())
        else:
            if not target_dict[target_dir.name]:
                del target_dict[target_dir.name]
            if VERBOSE:
                print(
                    f"Done, found {sum(len(target_list) for target_list in target_dict.values())} targets!"
                )
    else:
        # No directory, generating an image
        if VERBOSE:
            print(f"Generating a Ukraine Flag image")
        target_dict["target"].append(generate_ukrainian_flag())

    return target_dict


# =============================================================================
# IMAGE COMPOSITION
# =============================================================================


def get_random_item(candidates: list[Image.Image]) -> Image.Image:
    """
    Select a random item from the available options.

    Args:
        candidates: List of available images.

    Returns:
        A randomly selected image.
    """

    # Get random index in the list
    idx = np.random.randint(0, len(candidates))
    # return the item at the index
    return candidates[idx]


def get_obbox(
    trans_x: int,
    trans_y: int,
    theta: float,
    width: int,
    height: int,
    super_width: int,
    super_height: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Calculates the oriented bounding box of a target given its translation, size and orientation

    Args:
        trans_x (int): The amount the target was translated in the x direction
        trans_y (int): The amount the target was translated in the y direction
        theta (float): the amount the target was rotated in degrees
        width (int): the width of the target (post scaling)
        height (int): the height of the target (post scaling)
        super_width (int): the width of the target image (post scaling and rotating)
        super_height (int): the height of the target image (post scaling and rotating)

    Returns:
        tuple[float, float, float, float, float, float, float, float]: A bounding box in the form of x1,y1,x2,y2,x3,y3,x4,y4 where x1,y1 is the upper left corner and the following coordinates move clockwise around the image. Not yet normalized for yolo obb
    """
    # PIL rotates counter-clockwise, so we negate the angle
    theta_rad = np.deg2rad(-theta)

    # The center of the rotated image (super bounding box)
    cx = trans_x + super_width / 2
    cy = trans_y + super_height / 2

    # Corners of the unrotated rectangle relative to center:
    corners_rel = [
        (-width / 2, -height / 2),  # top-left
        (width / 2, -height / 2),  # top-right
        (width / 2, height / 2),  # bottom-right
        (-width / 2, height / 2),  # bottom-left
    ]

    rotated_corners = []
    for rx, ry in corners_rel:
        # Apply rotation
        new_x = rx * np.cos(theta_rad) - ry * np.sin(theta_rad)
        new_y = rx * np.sin(theta_rad) + ry * np.cos(theta_rad)
        # Translate to actual position
        rotated_corners.append((cx + new_x, cy + new_y))

    x1, y1 = rotated_corners[0]
    x2, y2 = rotated_corners[1]
    x3, y3 = rotated_corners[2]
    x4, y4 = rotated_corners[3]

    return (x1, y1, x2, y2, x3, y3, x4, y4)


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
    image: Image.Image, shift_range: tuple[int, int] = (-30, 30)
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
    image: Image.Image, factor_range: tuple[float, float] = (0.5, 1.5)
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
    image: Image.Image, num_lines_range: tuple[int, int] = (1, 5)
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
    image: Image.Image, angle_range: tuple[float, float] = (-15, 15)
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
    class_id: int,
    bbox: (
        tuple[float, float, float, float]
        | tuple[float, float, float, float, float, float, float, float]
    ),
) -> str:
    """
    Generate a YOLO format annotation string.

    Args:
        class_id: The class ID for the object (0 for TARGET).
        bbox: Bounding box normalized to [0, 1]. Either standard format
              (x_center, y_center, width, height) or OBB format
              (x1, y1, x2, y2, x3, y3, x4, y4) with corner coordinates.

    Returns:
        YOLO annotation string: "class_id coord1 coord2 ..." with all bbox values.
    """

    return f"{class_id}" + "".join(f" {coord}" for coord in bbox)


def get_next_image_index(obj_dir: Path) -> int:
    """
    Find the next available image index when appending to existing dataset.

    Scans the obj directory for existing images and returns the next index.

    Args:
        obj_dir: Path to the data/obj directory.

    Returns:
        Next available integer index for image naming.
    """

    next_ind = 0
    for file in os.scandir(obj_dir):
        if ".jpg" in file.name:
            next_ind += 1
    return next_ind


def save_image_and_annotation(
    image: Image.Image,
    annotation: str,
    index: int,
    obj_dir: Path = OBJ_DIR,
) -> Path:
    """
    Save an image and its corresponding YOLO annotation file.

    Args:
        image: The generated image to save.
        annotation: YOLO format annotation string.
        index: Image index for filename.
        obj_dir: Directory to save files to.

    Returns:
        Path to the saved image file.
    """

    file_path = obj_dir / f"img_{index}"

    image.save(f"{file_path}.jpg")

    with open(f"{file_path}.txt", "at") as text_file:
        text_file.write(annotation)

    return Path(f"{file_path}.jpg")


def update_data_files(
    image_paths: list[Path], training_split: int, append: bool = True
) -> None:
    """
    Update train.txt, valid.txt, and alldata.txt with new image paths.

    Args:
        image_paths: List of paths to the generated images.
        training_split: Percentage of images to use for training (0-100).
        append: If True, append to existing files. If False, overwrite.
    """

    num_images = len(image_paths)
    # we first make the valid set since it is likely to be the smallest of the two
    num_valid = num_images // (100 - training_split)

    valid_paths = random.sample(image_paths, num_valid)
    train_paths = list(set(image_paths) - set(valid_paths))

    if append:
        with open(ALLDATA_TXT, "at") as alldata:
            alldata.writelines(map(lambda path: str(path) + "\n", image_paths))
        with open(TRAIN_TXT, "at") as traindata:
            traindata.writelines(map(lambda path: str(path) + "\n", train_paths))
        with open(VALID_TXT, "at") as validdata:
            validdata.writelines(map(lambda path: str(path) + "\n", valid_paths))
    else:
        with open(ALLDATA_TXT, "wt") as alldata:
            alldata.writelines(map(lambda path: str(path) + "\n", image_paths))
        with open(TRAIN_TXT, "wt") as traindata:
            traindata.writelines(map(lambda path: str(path) + "\n", train_paths))
        with open(VALID_TXT, "wt") as validdata:
            validdata.writelines(map(lambda path: str(path) + "\n", valid_paths))


def update_data_yaml(class_names: Iterable[any] = ["TARGET"]) -> None:
    """
    Update or create the data.yaml configuration file.

    Args:
        class_names: List of class names.
    """
    # Load in current data.yaml
    with open(DATA_YAML, "r") as file:
        data_dict: dict = yaml.safe_load(file)

    # Make sure the data paths are in the yaml
    if not data_dict.get("train", None):
        data_dict["train"] = str(TRAIN_TXT)
    if not data_dict.get("val", None):
        data_dict["val"] = str(VALID_TXT)

    if not type(class_names) == list:
        class_names = list(class_names)
    # Set the class fields
    data_dict["nc"] = len(class_names)
    data_dict["names"] = class_names

    # Write the updated data.yaml
    with open(DATA_YAML, "wt") as file:
        yaml.dump(data_dict, file)


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

    text_files = ["alldata.txt", "train.txt", "valid.txt"]

    # Check if directory exists
    if os.path.isdir(OBJ_DIR):
        # Directory alreaedy exists, if clean then clear it out
        if clean:
            shutil.rmtree(OBJ_DIR)
            os.makedirs(OBJ_DIR)
            for fname in text_files:
                if os.path.isfile(DATA_DIR / fname):
                    os.remove(DATA_DIR / fname)
                os.close(os.open(DATA_DIR / fname, flags=os.O_CREAT | os.O_TRUNC))

    elif os.path.isdir(DATA_DIR):
        # obj/ doesn't exist, double check for data/
        os.makedirs(OBJ_DIR)
        for fname in text_files:
            if os.path.isfile(DATA_DIR / fname):
                os.remove(DATA_DIR / fname)
            os.close(os.open(DATA_DIR / fname, flags=os.O_CREAT | os.O_TRUNC))
    else:
        # Directories don't exist
        os.makedirs(OBJ_DIR)
        for fname in text_files:
            os.close(os.open(DATA_DIR / fname, flags=os.O_CREAT | os.O_TRUNC))

    if clean:
        with open(DATA_YAML, "wt") as file:
            yaml.dump({"train": str(TRAIN_TXT), "val": str(VALID_TXT)}, file)


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================


def generate_image(
    background: Image.Image,
    target: Image.Image,
    apply_augmentation: bool = True,
) -> tuple[Image.Image, tuple[float, float, float, float, float, float, float, float]]:
    """
    Generate a single synthetic training image.

    Args:
        background: The background Image.
        target: The target Image to overlay onto the background.
        apply_augmentation: Whether to apply random augmentations.

    Returns:
        Tuple of (generated_image, yolo_obbox) where yolo_obbox is an 8-value
        oriented bounding box (x1, y1, x2, y2, x3, y3, x4, y4) normalized to [0, 1].
    """

    # Scaling factor for scaling the image
    scale = np.random.normal(1.0, 0.1)

    # Angle for rotation
    rotation = np.random.randint(0, 360)

    new_size = (round(target.size[0] * scale), round(target.size[1] * scale))

    # scale and rotate
    oriented_target = target.resize(new_size, Image.Resampling.HAMMING).rotate(
        rotation, Image.Resampling.BILINEAR, True, fillcolor=(0, 0, 0, 0)
    )

    trans_x = np.random.randint(0, background.size[0] - oriented_target.size[0])
    trans_y = np.random.randint(0, background.size[1] - oriented_target.size[1])

    # Paste and translate the reorented target onto the background
    # the mask makes sure that we dont delete neighboring pixels due to the rotation
    # the .copy ensures that that background remains unmodified and a new image is created
    composition = background.copy()
    composition.paste(im=oriented_target, box=(trans_x, trans_y), mask=oriented_target)

    # Calculate corners in PIL coordinates
    obbox = get_obbox(
        trans_x,
        trans_y,
        rotation,
        new_size[0],
        new_size[1],
        oriented_target.size[0],
        oriented_target.size[1],
    )
    # normalize to 0-1 for yolo
    yolo_obbox = tuple(
        corner / background.size[i % 2] for i, corner in enumerate(obbox)
    )

    # TODO: apply a tranformation to give some more perspective to the target

    # if augment is enabled randomly choose to apply one or more

    return composition, yolo_obbox


def generate_dataset(
    num_images: int, training_split: int, clean: bool = False, num_bg: int = 5
) -> None:
    """
    Generate a complete dataset of synthetic training images.

    Args:
        num_images: Total number of images to generate.
        training_split: Percentage of images for training (rest for validation).
        clean: If True, clear existing data before generating.
        num_bg: Number of procedural backgrounds to generate if directory is empty.
    """

    if VERBOSE:
        print("Generating dataset...")

    # Ensure that the YOLO directory structure is set up
    setup_data_directory(clean)

    # Get a list of backgrounds
    backGrounds = load_background_images(BG_DIR, num_bg)

    # Get a list of targets
    targets = load_target_images(TARGET_DIR)

    update_data_yaml(targets.keys())

    starting_index = get_next_image_index(OBJ_DIR)
    data_list = []

    for class_id, target_list in enumerate(targets.values()):
        for idx in range(num_images):
            # Pick a background
            bg = get_random_item(backGrounds)
            # Pick a target
            target = get_random_item(target_list)

            # Put the target on the background
            gen, bbox = generate_image(bg, target)

            # TODO: add support for more than one type of class
            annotation = generate_yolo_annotation(class_id, bbox)

            # Save the image in the directory,
            # and save the path to list to later write alldata.txt train.txt and val.txt
            data_list.append(
                save_image_and_annotation(gen, annotation, starting_index + idx)
            )

    update_data_files(data_list, training_split, not clean)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the targetGenerator.

    Args:
        args: Parsed command line arguments.
    """
    if args.verbose:
        print(f"Target Generator")
        print(f"  Images to generate: {args.number}")
        print(f"  Training split: {args.training_split}%")
        print(f"  Clean mode: {args.clean}")
        print(f"  Image size: {args.width}x{args.height}")
        print(f"  Target directory: {args.target_dir}")
        print(f"  Background directory: {args.bg_dir}")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Procedural backgrounds: {args.num_bg}")
        print()

    generate_dataset(
        num_images=args.number,
        training_split=args.training_split,
        clean=args.clean,
        num_bg=args.num_bg,
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
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose mode, more information is printed in verbose mode",
        action="store_true",
    )
    parser.add_argument(
        "--width",
        help=f"Width of generated images in pixels (default: {DEFAULT_IMG_WIDTH})",
        default=DEFAULT_IMG_WIDTH,
        type=int,
    )
    parser.add_argument(
        "--height",
        help=f"Height of generated images in pixels (default: {DEFAULT_IMG_HEIGHT})",
        default=DEFAULT_IMG_HEIGHT,
        type=int,
    )
    parser.add_argument(
        "--target-dir",
        help=f"Directory containing target images (default: {TARGET_DIR})",
        default=TARGET_DIR,
        type=Path,
    )
    parser.add_argument(
        "--bg-dir",
        help=f"Directory containing background images (default: {BG_DIR})",
        default=BG_DIR,
        type=Path,
    )
    parser.add_argument(
        "--data-dir",
        help=f"Directory for output data (default: {DATA_DIR})",
        default=DATA_DIR,
        type=Path,
    )
    parser.add_argument(
        "--num-bg",
        help="Number of procedural backgrounds to generate if bg-dir is empty (default: 5)",
        default=5,
        type=int,
    )

    args = parser.parse_args()
    VERBOSE = args.verbose
    DEFAULT_IMG_WIDTH = args.width
    DEFAULT_IMG_HEIGHT = args.height
    TARGET_DIR = args.target_dir
    BG_DIR = args.bg_dir
    DATA_DIR = args.data_dir
    OBJ_DIR = DATA_DIR / "obj"
    TRAIN_TXT = DATA_DIR / "train.txt"
    VALID_TXT = DATA_DIR / "valid.txt"
    ALLDATA_TXT = DATA_DIR / "alldata.txt"

    main(args)
