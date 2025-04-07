# Assignment 3: Image Upscaling with Convolutional Autoencoder

**Author:** Zer0F8th
**Date:** April 7, 2025

## Overview

This project implements a Convolutional Autoencoder (CAE) using TensorFlow and Keras to perform image upscaling. The goal is to take low-resolution (28x28) images of cats and dogs as input and train the network to generate higher-resolution versions (configurable, default 56x56). This assignment focuses on the autoencoder architecture for reconstruction and dimensionality change, *not* on image classification.

[](images/upscale_viz_10890.png)

## Features

* Builds and trains a Convolutional Autoencoder specifically designed for upscaling image dimensions.
* Processes images from a structured dataset directory (`data/train/`, `data/test/`).
* Supports both Grayscale (`CHANNELS = 1`) and RGB (`CHANNELS = 3`) color modes (configurable in the script).
* Implements data caching using NumPy (`.npy`) files in the `processed_data/` directory to significantly speed up data loading on subsequent runs.
* Utilizes Python's `logging` module for detailed console output, tracking progress, warnings, and errors.
* Includes a visualization step that compares the low-resolution input, the autoencoder's upscaled output, and the high-resolution ground truth for a sample test image.
* Saves the generated visualization plot to a file in the `visualization_outputs/` directory.
* Saves the trained Keras model to an `.h5` file.

## Requirements

* Python 3.x
* TensorFlow (`pip install tensorflow`)
* NumPy (`pip install numpy`)
* Pillow (PIL Fork) (`pip install Pillow`)
* Matplotlib (`pip install matplotlib`)

Install the requirments from `requirements.txt`:


```bash
pip install -r requirments.txt
```


## Dataset Structure

The script expects the following directory structure in the same location as the Python script:

```
.
├── main.py                 # Main Script
├── data/                   # Needs to be created manually
│   ├── train/              # Contains training images (e.g., cat.0.jpg, dog.100.jpg)
│   └── test/               # Contains testing images (e.g., 1.jpg, 2.jpg)
└── README.md               # This file
```


* The specific filenames in `train/` (like `cat.` or `dog.`) are **not** used for labels in this assignment.
* The `test/` images are used for validation during training and for generating the final visualization.

## Setup

1.  **Clone/Download:** Obtain the project files (Python script).
2.  **Install Dependencies:** Open your terminal or command prompt and run:
    ```bash
    pip install -r requirments.txt
    ```
3.  **Create Data Directory:** Create the `data` directory in the same folder as the script.
4.  **Populate Data:** Place your training images (`.jpg`) into the `data/train/` subdirectory and your testing images (`.jpg`) into the `data/test/` subdirectory.

## Usage

1.  **Navigate:** Open your terminal or command prompt and navigate to the directory containing the script (`assignment3.py`).
2.  **Run:** Execute the script using Python:
    ```bash
    python main.py
    ```
    * **First Run:** The script will process all images in `data/train/` and `data/test/`, resize them, normalize them, and save them as `.npy` files in the `processed_data/` directory. This may take a significant amount of time depending on the dataset size and your hardware.
    * **Subsequent Runs:** The script will detect the cached `.npy` files in `processed_data/` and load them directly, skipping the image processing step and starting model building/training much faster.
3.  **Monitor:** Observe the console output for logs detailing the process, including data loading, model summary, training progress (loss/validation loss per epoch), and final visualization steps.

## Configuration

Key parameters can be adjusted directly within the Python script (`main.py`):

* `IMG_WIDTH_SMALL`, `IMG_HEIGHT_SMALL`: Dimensions of the low-resolution input images (Default: 28x28).
* `IMG_WIDTH_LARGE`, `IMG_HEIGHT_LARGE`: Target dimensions for the upscaled output images (Default: 56x56). **Note:** If you change this, you *must* verify that the decoder architecture in the script correctly outputs this shape (`model.summary()` is crucial).
* `CHANNELS`: Set to `1` for Grayscale processing or `3` for RGB processing. `COLOR_MODE_PIL` will update automatically. (Default: 1).
* `EPOCHS`: Number of training epochs (Default: 50). Start lower (e.g., 10-20) for initial testing.
* `BATCH_SIZE`: Number of samples per gradient update during training (Default: 32). Decrease if you encounter memory errors.
* Paths (`TRAIN_DATA_PATTERN`, `TEST_DATA_PATTERN`, etc.): Automatically determined relative to the script, usually don't need changing if the `data` directory is setup correctly.

## Output

The script will produce the following:

1.  **Console Logs:** Detailed information about the script's execution phases.
2.  **`processed_data/` directory:** Contains cached `.npy` files (e.g., `x_train_...npy`, `y_train_...npy`) for faster subsequent runs.
3.  **`visualization_outputs/` directory:** Contains the saved comparison plot (e.g., `upscale_viz_1.png`) generated after training, showing the input, generated output, and ground truth.
4.  **`.h5` Model File:** The trained Keras model saved (e.g., `upscaler_autoencoder_56x56_1ch.h5`).

## File Structure (After Running)

```
.
├── main.py                 # Main Python script
├── data/                   # Root data directory (needs to be created)
│   ├── train/              # Training images (e.g., cat.0.jpg, dog.0.jpg...)
│   └── test/               # Testing images (e.g., 1.jpg, 2.jpg...)
├── processed_data/         # Created automatically for cached data
│   ├── x_train_... .npy
│   ├── y_train_... .npy
│   ├── x_test_... .npy
│   └── y_test_... .npy
├── visualization_outputs/          # Created automatically for saved plots
│   └── upscale_viz_... .png
├── upscaler_autoencoder_... .h5    # Saved Keras model
└── README.md                       # This file
```

