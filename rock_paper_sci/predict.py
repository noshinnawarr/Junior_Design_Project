model = YOLO('/content/runs/classify/train/weights/best.pt')

results = model("/content/Rock-Paper-Scissors-1/train/scissors/scissors02-041_png.rf.ecf1e4b9286a6065ee10147348c4a84d.jpg") #, save=True, conf=0.5)
results[0].show()