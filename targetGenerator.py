"""This module is a fairly simple python module that will generate images of a target on a background, it will also generate annotated data files for use in training a yolo model."""

import os

import argparse
import pillow


def main():
    """Main entry point for the targetGenerator

    - sets up data directory and files
    - ingests inputs and building brick data
    - kicks off generation

    """
    
    # create data/obj directory
    # if the directory doesnt exist make it
    # otherwise if -c is specified delete the directory and then make it again(or clear it out)

    # pull in background images

    # pull in target images(or generate them)

    # for args.n generate an image and then store its YOLO annotation and add its path to train.txt

        #image generation
        #pick random background
        #pick random target
        #randomly place target on background
        #randomly apply augmentation(future add-on)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--clean", help="Cleans data directory and data.yaml", action="store_true")
    parser.add_argument("-n", "--number", help="Specifies the total number of images to generate", default=100, type=int)
    parser.add_argument("-ts","--training_split", help="Specifies the percentage of values which are put as training images, the rest are validation", default=100, type=int, choices=range(0,101))
    args = parser.parse_args()
    main()
