import os
import argparse
import joblib
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description ='Please provide a handwritten image to identify')
parser.add_argument('--file', '-f', help ='Path to image file')

args = parser.parse_args()
if os.path.exists(args.file):
	model = joblib.load("model_joblib")
	img = Image.open(args.file).convert('L')
	img = img.resize((8, 8), Image.LANCZOS)
	img_array = np.array(img)
	img_array = (img_array / 255.0) * 16.0
	img_array = img_array.flatten().reshape(1, -1)
	prediction = model.predict(img_array)
	print(prediction)
else:
	print('File doesnot exist')
