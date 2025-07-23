import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from skimage.feature import local_binary_pattern
'''
Parameters for LBP

Radius (r) = the radius of the circle which will account for different scales

# of points (p) =  in a circularly symmetric neighborhood to consider (thus removing relying on a square neighborhood)

'''

data_set = 'train'

radius = 2  
n_points =  8 * radius 

save_path = f'results/{data_set}'
save_filename = f'lbp_features_{data_set}.csv'
save_full_path = os.path.join(save_path, save_filename)

os.makedirs(save_path, exist_ok=True)

def compute_lbp_histogram(image, n_points, radius):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

# Lists to store feature vectors for each class
nevus_data = []
others_data = []

# Process Nevus images
nevus_images = glob.glob(f'{data_set}/nevus/*.jpg')
for img_path in tqdm(nevus_images, desc="Processing Nevus Images"):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    if img is not None:
        lbp_hist = compute_lbp_histogram(img, n_points, radius)
        nevus_data.append({
            "image_path": img_path,
            **{f"lbp_bin_{i}": lbp_hist[i] for i in range(len(lbp_hist))},
            "label": 0  # Nevus class label
        })
    else:
        print(f"Failed to load image {img_path}")

# Process Others images
others_images = glob.glob(f'{data_set}/others/*.jpg')
for img_path in tqdm(others_images,desc="Processing Others Images"):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    if img is not None:
        lbp_hist = compute_lbp_histogram(img, n_points, radius)
        others_data.append({
            "image_path": img_path,
            **{f"lbp_bin_{i}": lbp_hist[i] for i in range(len(lbp_hist))},
            "label": 1  # Others class label
        })
    else:
        print(f"Failed to load image {img_path}")

# Process Test images
if data_set == 'test':
    
    test_data = []
    
    test_images = glob.glob(f'{data_set}/*.jpg')
    for img_path in tqdm(test_images,desc="Processing Test Images"):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        if img is not None:
            lbp_hist = compute_lbp_histogram(img, n_points, radius)
            others_data.append({
                "image_path": img_path,
                **{f"lbp_bin_{i}": lbp_hist[i] for i in range(len(lbp_hist))},
            })
        else:
            print(f"Failed to load image {img_path}")

# Combine data from both classes
all_data = nevus_data + others_data + test_data

# Convert to DataFrame and save to CSV
df = pd.DataFrame(all_data)
df.to_csv(save_full_path, index=False)
print(f"LBP features saved to {save_full_path}")