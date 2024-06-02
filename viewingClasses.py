import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab

# Path to the dataset and annotations
dataset_path = "data"
anns_file_path = os.path.join(dataset_path, 'annotations.json')

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.load(f)

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']

# Filter images to only include those in batch_1/train
batch_folder = 'batch_1'
filtered_imgs = [img for img in imgs if batch_folder in img['file_name']]

# Summary statistics
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(filtered_imgs)

# Extract category and supercategory names
cat_names = []
super_cat_names = []
super_cat_ids = {}
super_cat_last_name = ''
nr_super_cats = 0

for cat_it in categories:
    cat_names.append(cat_it['name'])
    super_cat_name = cat_it['supercategory']
    if super_cat_name != super_cat_last_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        super_cat_last_name = super_cat_name
        nr_super_cats += 1

print('Number of super categories:', nr_super_cats)
print('Number of categories:', nr_cats)
print('Number of annotations:', nr_annotations)
print('Number of images:', nr_images)

# Count annotations per category
cat_histogram = np.zeros(nr_cats, dtype=int)
for ann in anns:
    cat_histogram[ann['category_id']] += 1

# Plot the histogram of annotations per category
f, ax = plt.subplots(figsize=(5, 15))
df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
df = df.sort_values('Number of annotations', ascending=False)
sns.set_color_codes("pastel")
sns.set(style="darkgrid")
plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df, label="Total", color="b")
plt.show()

# Obtain Exif orientation tag code
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

# Load dataset as a COCO object
coco = COCO(anns_file_path)

# Randomly select 12 images from batch_1/train
selected_imgs = random.sample(filtered_imgs, min(12, len(filtered_imgs)))

# Display images and corresponding annotations in a 3x4 grid
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for ax, img in zip(axes, selected_imgs):
    img_id = img['id']
    image_filename = img['file_name']
    
    # Find batch folder from image filename
    image_path = os.path.join(dataset_path, image_filename)
    
    # Load image
    I = Image.open(image_path)
    
    # Load and process image metadata
    if I._getexif():
        exif = dict(I._getexif().items())
        if orientation in exif:
            if exif[orientation] == 3:
                I = I.rotate(180, expand=True)
            elif exif[orientation] == 6:
                I = I.rotate(270, expand=True)
            elif exif[orientation] == 8:
                I = I.rotate(90, expand=True)
    
    # Show image
    ax.axis('off')
    ax.imshow(I)
    
    # Load and show annotations
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
