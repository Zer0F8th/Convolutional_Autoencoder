import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import glob
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
import logging 
import time   

# --- Logging Configuration ---
# Logs will be shown in the console.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get script directory
LOG_DIR = os.path.join(BASE_DIR, 'logs') # Directory for logs
os.makedirs(LOG_DIR, exist_ok=True) # Create log directory if it doesn't exist
logging.basicConfig(filename='logs/run.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Script started.")
logging.info(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}") # Add current time

# --- Configuration ---
IMG_WIDTH_SMALL = 28
IMG_HEIGHT_SMALL = 28
IMG_WIDTH_LARGE = 56  # Target output width (at least 56)
IMG_HEIGHT_LARGE = 56 # Target output height (at least 56)

CHANNELS = 1 # 1 for Grayscale, 3 for RGB
COLOR_MODE_PIL = 'L' if CHANNELS == 1 else 'RGB'
logging.info(f"Image settings: Input={IMG_WIDTH_SMALL}x{IMG_HEIGHT_SMALL}, Output={IMG_WIDTH_LARGE}x{IMG_HEIGHT_LARGE}, Channels={CHANNELS} ({COLOR_MODE_PIL})")

# --- Paths ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DATA_PATTERN = os.path.join(DATA_DIR, 'train', '*.jpg')
TEST_DATA_PATTERN = os.path.join(DATA_DIR, 'test', '*.jpg')

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualization_outputs') # Directory for saved plots

# Define specific cache file paths
X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, f'x_train_{IMG_WIDTH_SMALL}x{IMG_HEIGHT_SMALL}_{CHANNELS}ch.npy')
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, f'y_train_{IMG_WIDTH_LARGE}x{IMG_HEIGHT_LARGE}_{CHANNELS}ch.npy')
X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, f'x_test_{IMG_WIDTH_SMALL}x{IMG_HEIGHT_SMALL}_{CHANNELS}ch.npy')
Y_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, f'y_test_{IMG_WIDTH_LARGE}x{IMG_HEIGHT_LARGE}_{CHANNELS}ch.npy')

SAVED_MODEL_PATH = os.path.join(BASE_DIR, f'upscaler_autoencoder_{IMG_WIDTH_LARGE}x{IMG_HEIGHT_LARGE}_{CHANNELS}ch.h5')

# --- Hyperparameters ---
EPOCHS = 50 # Start lower (e.g., 10-20) to test
BATCH_SIZE = 32
logging.info(f"Training Hyperparameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}")

# --- Create necessary directories ---
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
logging.info(f"Ensured directories exist: '{PROCESSED_DATA_DIR}', '{VISUALIZATION_DIR}'")

# --- Data Loading and Preprocessing Function ---
def load_and_process_data(image_paths, target_small_size, target_large_size, color_mode='L'):
    """Loads images, resizes them for input (small) and target (large), and normalizes."""
    data_small = []
    data_large = []
    num_images = len(image_paths)
    logging.info(f"Processing {num_images} images...")

    target_w_small, target_h_small = target_small_size
    target_w_large, target_h_large = target_large_size
    num_channels = 1 if color_mode == 'L' else 3

    for i, filepath in enumerate(image_paths):
        try:
            image = Image.open(filepath).convert(color_mode)

            # Create small input version
            image_small = image.resize(target_small_size, Image.Resampling.LANCZOS)
            data_small.append(np.array(image_small))

            # Create large target version (resize original)
            image_large = image.resize(target_large_size, Image.Resampling.LANCZOS)
            data_large.append(np.array(image_large))

            if (i + 1) % 1000 == 0 or (i + 1) == num_images:
                logging.info(f"  Processed {i + 1}/{num_images}")

        except FileNotFoundError:
             logging.error(f"File not found: {filepath}. Skipping.")
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}. Skipping.")

    logging.info("Converting to NumPy arrays and normalizing...")
    try:
        data_small = np.array(data_small, dtype=np.float32)
        data_large = np.array(data_large, dtype=np.float32)

        # Normalize pixel values to [0, 1]
        data_small /= 255.0
        data_large /= 255.0

        # Reshape for Keras (add channel dimension)
        data_small = np.reshape(data_small, (len(data_small), target_h_small, target_w_small, num_channels))
        data_large = np.reshape(data_large, (len(data_large), target_h_large, target_w_large, num_channels))

        logging.info(f"Finished processing. Shapes: Small={data_small.shape}, Large={data_large.shape}")
        return data_small, data_large
    except ValueError as ve:
        logging.error(f"Error during NumPy conversion/reshape: {ve}. This might happen if some images failed to load.")
        logging.error("Check previous error messages. Returning None.")
        return None, None


# --- Load or Process Data with Caching ---

# Function to handle loading/processing for a dataset split (train/test)
def get_data(split_name, pattern, x_cache_path, y_cache_path, small_size, large_size, color_mode_pil):
    """Loads data from cache if available, otherwise processes and saves to cache."""
    logging.info(f"--- Handling {split_name} data ---")
    logging.info(f"Checking for cached data: X='{os.path.basename(x_cache_path)}', Y='{os.path.basename(y_cache_path)}'")

    if os.path.exists(x_cache_path) and os.path.exists(y_cache_path):
        logging.info(f"Cached {split_name} data found. Loading...")
        try:
            x_data = np.load(x_cache_path)
            y_data = np.load(y_cache_path)
            logging.info(f"Loaded {split_name} data: X shape={x_data.shape}, Y shape={y_data.shape}")
            return x_data, y_data
        except Exception as e:
            logging.error(f"Error loading cached {split_name} data: {e}. Reprocessing.")
            # Attempt to remove potentially corrupted cache files
            if os.path.exists(x_cache_path): os.remove(x_cache_path)
            if os.path.exists(y_cache_path): os.remove(y_cache_path)

    logging.info(f"Cached {split_name} data not found or failed to load. Processing images from pattern: {pattern}")
    image_files = glob.glob(pattern)
    if not image_files:
        logging.warning(f"No images found for pattern: {pattern}")
        return None, None

    x_data, y_data = load_and_process_data(
        image_files,
        small_size,
        large_size,
        color_mode_pil
    )

    if x_data is not None and y_data is not None:
        logging.info(f"Saving processed {split_name} data to cache...")
        try:
            np.save(x_cache_path, x_data)
            np.save(y_cache_path, y_data)
            logging.info(f"Saved {split_name} data cache successfully.")
        except Exception as e:
            logging.error(f"Error saving {split_name} data to cache: {e}")
    else:
        logging.warning(f"Processing failed for {split_name} data. Cache not saved.")

    return x_data, y_data

# Get Training Data
x_train, y_train = get_data(
    "training",
    TRAIN_DATA_PATTERN,
    X_TRAIN_PATH, Y_TRAIN_PATH,
    (IMG_WIDTH_SMALL, IMG_HEIGHT_SMALL),
    (IMG_WIDTH_LARGE, IMG_HEIGHT_LARGE),
    COLOR_MODE_PIL
)

# Get Test Data
x_test, y_test = get_data(
    "test",
    TEST_DATA_PATTERN,
    X_TEST_PATH, Y_TEST_PATH,
    (IMG_WIDTH_SMALL, IMG_HEIGHT_SMALL),
    (IMG_WIDTH_LARGE, IMG_HEIGHT_LARGE),
    COLOR_MODE_PIL
)

# Exit if training data loading failed
if x_train is None or y_train is None:
    logging.critical("Essential training data (x_train or y_train) could not be loaded or processed. Exiting.")
    exit()


# --- Build the Autoencoder Model ---
logging.info("Building the Autoencoder Model...")
input_img = Input(shape=(IMG_HEIGHT_SMALL, IMG_WIDTH_SMALL, CHANNELS))

# Encoder (28x28 -> 14x14 -> 7x7)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv1')(input_img)
x = MaxPooling2D((2, 2), padding='same', name='enc_pool1')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv2')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='enc_pool2')(x)
logging.info(f"Encoder output shape (latent space): {encoded.shape}")

# Decoder (7x7 -> 14x14 -> 28x28 -> 56x56)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv1')(encoded)
x = UpSampling2D((2, 2), name='dec_upsample1')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv2')(x)
x = UpSampling2D((2, 2), name='dec_upsample2')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='dec_conv3')(x)
x = UpSampling2D((2, 2), name='dec_upsample3')(x)
decoded = Conv2D(CHANNELS, (3, 3), activation='sigmoid', padding='same', name='output_conv')(x)
logging.info(f"Decoder output shape: {decoded.shape}")

autoencoder = Model(input_img, decoded, name='UpscalingAutoencoder')

# --- Compile the Model ---
logging.info("Compiling the model...")
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # or 'mse'

logging.info("\n" + "="*30 + " Model Summary " + "="*30)
autoencoder.summary(print_fn=logging.info) # Log summary directly
logging.info("="*75)

# Verify output shape match
if autoencoder.output_shape[1:] != y_train.shape[1:]:
     logging.critical("\n\n***** OUTPUT SHAPE MISMATCH *****")
     logging.critical(f"Model output shape {autoencoder.output_shape[1:]} != Y data shape {y_train.shape[1:]}!")
     logging.critical("Adjust decoder architecture or target image size and REPROCESS data.")
     exit()
else:
    logging.info("Model output shape matches target data shape.")


# --- Train the Model ---
logging.info("Preparing for training...")

# Prepare validation data
validation_data = None
if x_test is not None and y_test is not None:
    validation_data = (x_test, y_test)
    logging.info(f"Using Test data for validation: x_test={x_test.shape}, y_test={y_test.shape}")
elif x_train is not None and y_train is not None and len(x_train) > 10: # Check if enough data for split
    # Fallback: Use a portion of training data for validation
    split_ratio = 0.1
    split_idx = int(len(x_train) * (1.0 - split_ratio))
    if split_idx > 0 and len(x_train) - split_idx > 0:
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train_new = x_train[:split_idx]
        y_train_new = y_train[:split_idx]
        # Check if split resulted in non-empty arrays before reassigning
        if x_train_new.size > 0 and x_val.size > 0:
             x_train, y_train = x_train_new, y_train_new # Reassign only if split is valid
             validation_data = (x_val, y_val)
             logging.warning("No test data found. Using last 10% of training data for validation.")
             logging.info(f"New training shape: x_train={x_train.shape}, y_train={y_train.shape}")
             logging.info(f"Validation shape: x_val={x_val.shape}, y_val={y_val.shape}")
        else:
             logging.warning("Could not split training data for validation (too small). Training without validation split.")
    else:
         logging.warning("Training data too small to split for validation. Training without validation split.")
else:
    logging.warning("No test data available and training data insufficient or unavailable for validation split.")


logging.info("\n--- Starting Training ---")
start_time = time.time()
history = autoencoder.fit(x_train, y_train,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          validation_data=validation_data,
                          verbose=2) # verbose=2 shows one line per epoch
end_time = time.time()
logging.info(f"--- Training Finished --- (Duration: {end_time - start_time:.2f} seconds)")

# --- Save the Model ---
logging.info(f"Saving trained model to {SAVED_MODEL_PATH}...")
try:
    autoencoder.save(SAVED_MODEL_PATH)
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Error saving model: {e}")

# --- Visualization Example Function ---
def visualize_upscaling(model_path, test_image_path, output_dir):
    """Loads model, predicts on test image, shows and saves the comparison plot."""
    logging.info(f"\n--- Visualizing Results ---")
    logging.info(f"Model: {os.path.basename(model_path)}")
    logging.info(f"Test image: {os.path.basename(test_image_path)}")

    if not os.path.exists(test_image_path):
        logging.error(f"Visualization failed: Test image not found at {test_image_path}")
        return

    try:
        model = load_model(model_path)
        logging.info("Loaded model for visualization.")
    except Exception as e:
        logging.error(f"Error loading model '{model_path}': {e}")
        return

    # Get model input/output shapes dynamically
    try:
        input_h, input_w, input_c = model.input_shape[1:]
        output_h, output_w, output_c = model.output_shape[1:]
        current_color_mode = 'L' if input_c == 1 else 'RGB'
    except Exception as e:
        logging.error(f"Could not determine model input/output shapes: {e}")
        return

    # --- Load and Prepare Images ---
    try:
        # Load original test image
        orig_image = Image.open(test_image_path).convert(current_color_mode)

        # Prepare small input image for prediction
        input_pil = orig_image.copy().resize((input_w, input_h), Image.Resampling.LANCZOS)
        input_array = np.array(input_pil, dtype=np.float32) / 255.0
        # Add batch dimension
        input_array_batch = np.reshape(input_array, (1, input_h, input_w, input_c))

        # Prepare large ground truth image
        truth_pil = orig_image.copy().resize((output_w, output_h), Image.Resampling.LANCZOS)

    except Exception as e:
        logging.error(f"Error loading/preparing image {test_image_path}: {e}")
        return

    # --- Predict ---
    logging.info("Generating prediction...")
    try:
        generated_output = model.predict(input_array_batch)
    except Exception as e:
        logging.error(f"Error during model prediction: {e}")
        return

    # --- Post-process and Display ---
    try:
        # Remove batch dimension and ensure correct shape/type for display
        generated_image_array = np.reshape(generated_output, (output_h, output_w, output_c))
        # Clip values just in case they are slightly outside [0, 1] and scale back to [0, 255]
        generated_image_array_uint8 = np.clip(generated_image_array * 255.0, 0, 255).astype(np.uint8)

        # Use Pillow to handle potential channel issues for display
        if output_c == 1:
             # Remove channel dim for grayscale if it exists
            generated_pil = Image.fromarray(np.squeeze(generated_image_array_uint8), mode='L')
        else:
            generated_pil = Image.fromarray(generated_image_array_uint8, mode='RGB')

        logging.info("Preparing visualization plot...")
        fig, axes = plt.subplots(1, 3, figsize=(12, 5)) # Adjust figure size
        cmap_val = 'gray' if current_color_mode == 'L' else None

        # Input Image (Resized Small)
        axes[0].set_title(f"Input ({input_w}x{input_h})")
        axes[0].imshow(input_pil, cmap=cmap_val)
        axes[0].axis('off')

        # Generated Image (Upscaled)
        axes[1].set_title(f"Generated ({output_w}x{output_h})")
        axes[1].imshow(generated_pil, cmap=cmap_val)
        axes[1].axis('off')

        # Ground Truth Image (Resized Large)
        axes[2].set_title(f"Ground Truth ({output_w}x{output_h})")
        axes[2].imshow(truth_pil, cmap=cmap_val)
        axes[2].axis('off')

        plt.tight_layout()

        # --- Save the plot ---
        plot_filename = f"upscale_viz_{os.path.splitext(os.path.basename(test_image_path))[0]}.png"
        plot_save_path = os.path.join(output_dir, plot_filename)
        try:
            plt.savefig(plot_save_path)
            logging.info(f"Saved visualization plot to: {plot_save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot to {plot_save_path}: {e}")

        plt.show() # Show the plot interactively
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        logging.error(f"Error during plot generation or saving: {e}")
        # Attempt to close plot if it exists
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)


# --- Run Visualization (if model and test images exist) ---
if os.path.exists(SAVED_MODEL_PATH):
    test_files_viz = glob.glob(TEST_DATA_PATTERN)
    if test_files_viz:
         # Choose a random test image
         random_test_image = np.random.choice(test_files_viz)
         visualize_upscaling(SAVED_MODEL_PATH, random_test_image, VISUALIZATION_DIR)
    else:
         logging.warning("Model trained and saved, but no test images found for visualization.")
else:
    logging.warning("Trained model file not found. Skipping visualization.")


logging.info("Script finished.")