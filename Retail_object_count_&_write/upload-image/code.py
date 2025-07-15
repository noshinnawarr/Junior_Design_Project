from google.colab import files

uploaded = files.upload()
image_paths = list(uploaded.keys())
print("Uploaded images:", image_paths)
