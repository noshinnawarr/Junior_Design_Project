import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import sys

IMG_SIZE = 180
DEFAULT_MODEL = Path(__file__).parent / 'cat_dog_classifier.h5'

def load_and_prepare_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Failed to open image: {e}")
        sys.exit(1)

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path, model_path=None):
    # Import Keras here to avoid slowing our script unnecessarily (like in main.py)
    from tensorflow import keras

    model_file = Path(model_path) if model_path else DEFAULT_MODEL

    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        sys.exit(1)

    print(f"🔍 Loading model from: {model_file}")
    model = keras.models.load_model(model_file)

    print(f"🖼️  Preparing image: {image_path}")
    img = load_and_prepare_image(image_path)

    print("🤖 Running prediction...")
    prediction = model.predict(img)[0][0]

    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"\n✅ Prediction: {label}")
    print(f"🔢 Confidence Score: {confidence:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Cat vs Dog Image Classifier")
    parser.add_argument('-i', '--image', required=True, help="Path to image file")
    parser.add_argument('-m', '--model', help="Optional path to trained model (default: cat_dog_classifier.h5)")

    args = parser.parse_args()

    image_file = Path(args.image)
    if not image_file.exists():
        print(f"❌ Image file not found: {image_file}")
        sys.exit(1)

    predict(args.image, args.model)

if __name__ == "__main__":
    main()
