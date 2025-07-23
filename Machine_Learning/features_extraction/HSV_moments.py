import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

data_set = 'train'

save_path = f'results/{data_set}'
save_filename = f'HSV_features_with_moments_{data_set}.csv'
save_full_path = os.path.join(save_path, save_filename)

os.makedirs(save_path, exist_ok=True)

# Function to compute color histogram features in HSV
def compute_color_histogram(image, bins=8):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Compute histograms for H, S, V channels
    hist_h = cv.calcHist([hsv_image], [0], None, [bins], [0, 180]).flatten()  # Hue channel
    hist_s = cv.calcHist([hsv_image], [1], None, [bins], [0, 256]).flatten()  # Saturation channel
    hist_v = cv.calcHist([hsv_image], [2], None, [bins], [0, 256]).flatten()  # Value channel
    
    color_hist_sum = np.concatenate([hist_h, hist_s, hist_v])
    color_hist_sum = color_hist_sum / color_hist_sum.sum()  # Normalize to sum to 1
    
    return color_hist_sum

# Function to compute color moments (mean, standard deviation, skewness) for HSV
def compute_color_moments(image):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    channels = cv.split(hsv_image)
    
    # Calculate moments for each channel
    moments = []
    for channel in channels:
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean((channel - mean) ** 3) / (std ** 3) if std != 0 else 0
        moments.extend([mean, std, skewness])
        
    return moments

# Lists to store data for each class
nevus_data = []
others_data = []

# Process Nevus images
nevus_images = glob.glob(f'{data_set}/nevus/*.jpg')
for img_path in tqdm(nevus_images, desc="Processing Nevus Images"):
    img = cv.imread(img_path)
    
    if img is not None:
        color_hist_sum = compute_color_histogram(img, bins=8)
        color_moments = compute_color_moments(img)
        
        # Combine histogram and color moments into a single dictionary
        nevus_data.append({
            "image_path": img_path,
            **{f"color_hist_{i}": color_hist_sum[i] for i in range(len(color_hist_sum))},
            **{f"moment_{i}": color_moments[i] for i in range(len(color_moments))},
            "label": 0  # Nevus class label
        })
    else:
        print(f"Failed to load image {img_path}")

# Process Others images
others_images = glob.glob(f'{data_set}/others/*.jpg')
for img_path in tqdm(others_images, desc="Processing Others Images"):
    img = cv.imread(img_path)
    
    if img is not None:
        color_hist_sum = compute_color_histogram(img, bins=8)
        color_moments = compute_color_moments(img)
        
        # Combine histogram and color moments into a single dictionary
        others_data.append({
            "image_path": img_path,
            **{f"color_hist_{i}": color_hist_sum[i] for i in range(len(color_hist_sum))},
            **{f"moment_{i}": color_moments[i] for i in range(len(color_moments))},
            "label": 1  # Others class label
        })
    else:
        print(f"Failed to load image {img_path}")

# Combine data from both classes and save to CSV
all_data = nevus_data + others_data
df = pd.DataFrame(all_data)
df.to_csv(save_full_path, index=False)
print(f"HSV color features with moments saved to {save_full_path}")
