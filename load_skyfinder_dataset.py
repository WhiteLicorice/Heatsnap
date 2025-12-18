from __future__ import annotations
from pathlib import Path
import tensorflow as tf
import pandas as pd

def load_skyfinder_dataset(
    csv_path: Path,
    batch_size:int=32,
    img_size:tuple[int, int]=(224, 224), 
    shuffle:bool=True,
    use_cache:bool=True,
):
    # --- Performance Optimization ---
    # This allows the map function to ignore the order of files,
    # letting 16 cores finish tasks as fast as they can.
    options = tf.data.Options()
    options.deterministic = False 
    
    df = pd.read_csv(csv_path)
    img_root = "data/skyfinder_images/"
    
    paths = [f"{img_root}{int(row.camera_id)}/{row.filename}" for _, row in df.iterrows()]
    metadata = df[['day_of_year', 'hour', 'latitude', 'longitude', 'solar_elevation']].values.astype("float32")
    targets = df['heat_index'].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, metadata, targets))
    ds = ds.with_options(options)

    # 1. Shuffle the "Pointers" (Fast & Efficient)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    def process_row(path, meta, target):
        img_raw = tf.io.read_file(path)
        # Use fast_dev_run logic: decode_jpeg is faster than decode_image
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, img_size)
        return {"image": img, "metadata": meta}, target

    # 2. Parallel Load & Decode
    # We use 16 cores (AUTOTUNE) to saturate the CPU.
    ds = ds.map(process_row, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Caching
    # If the dataset is > 10GB, use .cache("data/sky_cache") to avoid RAM overflow
    if use_cache:
        ds = ds.cache() 

    # 4. Batch and Prefetch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE) # GPU never waits for CPU
    
    return ds