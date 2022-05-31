"""
Team-name:    changed later
Team-motto:   Memelords know better
Team-members: Florian Eder
              Moritz Enderle
              John Tran
"""

import json
import os
import skimage.io
import cv2

tiffs = []
for file in os.listdir("../public_data/"):
    if file.endswith(".tiff"):
        tiffs.append(file)

img_width = img_height = 257
width = 30 / img_width
height = 20 / img_height

for filename in tiffs:
    label_filename = "../public_data/" + filename.replace('.tiff', '.json')
    if os.path.exists(label_filename):
        with open(label_filename, 'r', encoding="utf-8") as file:
            label = json.loads(file.read())
    else:
        continue
    present = label["coordinates"]["present"]
    missing = label["coordinates"]["missing"]
    with open(f"../public_data/{filename.replace('.tiff', '.txt')}", "w", encoding="utf-8") as fo:
        for x, y in present:
            fo.write(f"1 {x / img_width} {(257 - y) / img_height} {width} {height}\n")
        for x, y in missing:
            fo.write(f"0 {x / img_width} {(257 - y) / img_height} {width} {height}\n")

    img_array = skimage.io.imread(f'../public_data/{filename}')
    img_gray = cv2.cvtColor(img_array * 100, cv2.COLOR_BGR2GRAY).astype('uint8')
    cv2.imwrite(f"../public_data/{filename.replace('.tiff', '.png')}", img_gray)

# to train the model, download the yolov5 github repo and run the following command:
# python3 train --data pills.yaml --img-size 256 --batch-size 16 --nosave --epochs 1000 --weights
#   yolov5n.pt --cfg yolov5n.yaml --names pills --device 0
