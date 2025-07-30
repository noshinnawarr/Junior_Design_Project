import cv2
import numpy as np
import pandas as pd
import os

def run(dataset_path, image_relative_path):
    # === Build image path
    img_path = os.path.abspath(os.path.join(dataset_path, image_relative_path))
    print(f"🔍 Checking image at: {img_path}")
    if not os.path.exists(img_path):
        print("❌ Image file not found:", image_relative_path)
        return

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image.")
        return

    # === Load colors.csv from same folder as script
    csv_path = os.path.join(os.path.dirname(__file__), 'colors.csv')
    if not os.path.exists(csv_path):
        print(f"❌ colors.csv not found at: {csv_path}")
        return

    index = ["color", "color_name", "hex", "R", "G", "B"]
    csv = pd.read_csv(csv_path, names=index, header=None)

    # === Declare state
    clicked = False
    r = g = b = xpos = ypos = 0

    # === Color matching function
    def getColorName(R, G, B):
        minimum = 10000
        cname = "Unknown"
        for i in range(len(csv)):
            d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
            if d <= minimum:
                minimum = d
                cname = csv.loc[i, "color_name"]
        return cname

    # === Mouse callback
    def draw_function(event, x, y, flags, param):
        nonlocal b, g, r, xpos, ypos, clicked
        if event == cv2.EVENT_LBUTTONDBLCLK:
            clicked = True
            xpos = x
            ypos = y
            b, g, r = img[y, x]
            b = int(b)
            g = int(g)
            r = int(r)

    # === Setup window and callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_function)

    # === Display loop
    while True:
        cv2.imshow("image", img)
        if clicked:
            cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
            text = getColorName(r, g, b) + f' R={r} G={g} B={b}'
            color_text = (0, 0, 0) if (r + g + b >= 600) else (255, 255, 255)
            cv2.putText(img, text, (50, 50), 2, 0.8, color_text, 2, cv2.LINE_AA)
            clicked = False

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
