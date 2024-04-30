# Using the Taco DataSet

## Instructions to Run
Download Annotations file from https://github.com/pedropro/TACO/tree/master/data

## Run
```
python downloadingOfficialImages.py --dataset_path <PATH_TO_ANNOATIONS_FILE>
```
Then proceed to run playaround.ipynb
 - Data splitting can be augmented by changing the trainTestValCumulativeSplit variable

Data can now be visualized by running viewingClasses.py
 - Includes a COCO-formatted bounding box and mask for a demo image from batch_1
