import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os
from skimage.feature import hog
from tqdm import tqdm

'''
Parameters for HOG

orientacion: Divides 180 degrees on the 'int' choose. This captures the direction of the edges:

pixels per cell: Defines the sixe of each cell in pixels and each cell with compute a histogram of gradient orientations

cells_per_block: This parameter defines the number of cells in each block, where normalization occurs. Normalizing across 
a block of cells helps reduce the effect of lighting and shadowing changes across the image

block_norm: Normalization method
'''

data_set = 'train'

hog_params = {
    "orientations": 8,  
    "pixels_per_cell": (16, 16),  
    "cells_per_block": (2, 2),  
    "block_norm": "L2-Hys",  
    "feature_vector": True  # Return as a 1D array
}

save_path = f'results/{data_set}'
save_filename = f'hog_features{data_set}.csv'
save_full_path = os.path.join(save_path, save_filename)

os.makedirs(save_path, exist_ok=True)

# Function to calculate HOG features
def compute_hog_features(image, hog_params):
    return hog(image, **hog_params)

# Lists to store feature vectors for each class
nevus_data = []
others_data = []

# Process Nevus images
nevus_images = glob.glob(f'{data_set}/nevus/*.jpg')
for img_path in tqdm(nevus_images):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv.resize(img, (256,256))
    
    if img is not None:
        hog_features = compute_hog_features(img, hog_params)
        nevus_data.append({
            "image_path": img_path,
            **{f"hog_{i}": hog_features[i] for i in range(len(hog_features))},
            "label": 0  # Nevus class label
        })
    else:
        print(f"Failed to load image {img_path}")

# Process Others images
others_images = glob.glob(f'{data_set}/others/*.jpg')
for img_path in tqdm(others_images):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv.resize(img, (256,256))
    
    if img is not None:
        hog_features = compute_hog_features(img, hog_params)
        others_data.append({
            "image_path": img_path,
            **{f"hog_{i}": hog_features[i] for i in range(len(hog_features))},
            "label": 1  # Others class label
        })
    else:
        print(f"Failed to load image {img_path}")

# Combine data from both classes
all_data = nevus_data + others_data

# Convert to DataFrame and save to CSV
df = pd.DataFrame(all_data)
df.to_csv(save_full_path, index=False)
print(f"HOG features saved to {save_full_path}")
