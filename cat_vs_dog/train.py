import os
import tensorflow as tf
import argparse
from pathlib import Path

IMG_SIZE = 180
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to 180x180
    image = tf.cast(image, tf.float32) / 255.0            # Normalize to [0,1]
    return image, label

def load_custom_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Custom dataset path not found: {path}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path / 'train',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        path / 'val',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # Normalize
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

def load_default_dataset():
    import tensorflow_datasets as tfds

    (train_ds, val_ds), ds_info = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True,
    )
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def train(dataset_path=None, output_path="cat_dog_classifier.h5"):
    from tensorflow.keras import layers, models
    
    print("Preparing dataset...")
    if dataset_path:
        train_ds, val_ds = load_custom_dataset(dataset_path)
    else:
        train_ds, val_ds = load_default_dataset()

    print("Building model...")
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("Training model...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=5)

    model.save(str(output_path))
    print(f"✅ Model saved as '{output_path}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Path to custom dataset folder (with 'train' and 'val' subfolders)")
    parser.add_argument('-o', '--output', help="Path to save the trained model (e.g., models/model1.h5)")
    args = parser.parse_args()

    train(args.dataset, output_path=args.output or "cat_dog_classifier.h5")

if __name__ == "__main__":
    main()
