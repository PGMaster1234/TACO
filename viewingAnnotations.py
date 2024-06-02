import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random

dataset_path = "data"
anns_file_path = os.path.join(dataset_path, 'annotations.json')
batch_folder = 'batch_1'

with open(anns_file_path, 'r') as f:
    dataset = json.load(f)

imgs = dataset['images']

filtered_imgs = [img for img in imgs if f'{batch_folder}/' in img['file_name']]

coco = COCO(anns_file_path)

selected_imgs = random.sample(filtered_imgs, min(24, len(filtered_imgs)))

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

fig, axes = plt.subplots(4, 6, figsize=(20, 20))
axes = axes.flatten()

for ax, img in zip(axes, selected_imgs):
    img_id = img['id']
    stored_image_filename = img['file_name']
    
    for subfolder in ['train', 'test', 'valid']:
        image_path = os.path.join(dataset_path, stored_image_filename)
        if os.path.exists(image_path):
            break
    else:
        print(f'Image not found in any folder: {stored_image_filename}')
        continue
    
    I = Image.open(image_path)
    
    if I._getexif():
        exif = dict(I._getexif().items())
        if orientation in exif:
            if exif[orientation] == 3:
                I = I.rotate(180, expand=True)
            elif exif[orientation] == 6:
                I = I.rotate(270, expand=True)
            elif exif[orientation] == 8:
                I = I.rotate(90, expand=True)
    
    ax.axis('off')
    ax.imshow(I)
    
    annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
    anns_sel = coco.loadAnns(annIds)
    
    for ann in anns_sel:
        color = colorsys.hsv_to_rgb(np.random.random(), 1, 1)
        for seg in ann['segmentation']:
            poly = Polygon(np.array(seg).reshape((int(len(seg) / 2), 2)))
            p = PatchCollection([poly], facecolor=color, edgecolors=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        [x, y, w, h] = ann['bbox']
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', alpha=0.7, linestyle='--')
        ax.add_patch(rect)

plt.tight_layout()
plt.show()
