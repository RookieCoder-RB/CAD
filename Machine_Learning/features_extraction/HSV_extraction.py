import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

data_set = 'test'

save_path = f'results/{data_set}'
save_filename = f'HSV_features_{data_set}.csv'
save_full_path = os.path.join(save_path, save_filename)

os.makedirs(save_path, exist_ok=True)

# Function to calculate HSV color histogram features
def compute_hsv_histogram(image, bins=8):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Calculate histogram for each channel (Hue, Saturation, Value)
    hist_h = cv.calcHist([hsv_image], [0], None, [bins], [0, 180]).flatten()  # Hue channel
    hist_s = cv.calcHist([hsv_image], [1], None, [bins], [0, 256]).flatten()  # Saturation channel
    hist_v = cv.calcHist([hsv_image], [2], None, [bins], [0, 256]).flatten()  # Value channel
    
    # Concatenate and normalize the histograms to create a single feature vector
    hsv_hist = np.concatenate([hist_h, hist_s, hist_v])
    hsv_hist = hsv_hist / hsv_hist.sum()  # Normalize to sum to 1
    
    return hsv_hist

# Lists to store data for each class
nevus_data = []
others_data = []
nevus_hist_sum = np.zeros(24) 
others_hist_sum = np.zeros(24)

# Process Nevus images
nevus_images = glob.glob(f'{data_set}/nevus/*.jpg')
for img_path in tqdm(nevus_images, desc="Processing Nevus Images"):
    
    img = cv.imread(img_path)
    
    if img is not None:
        color_hist_sum = compute_hsv_histogram(img, bins=8)
        nevus_data.append({
            "image_path": img_path,
            **{f"color_hist_{i}": color_hist_sum[i] for i in range(len(color_hist_sum))},
            "label": 0  # Nevus class label
        })
        nevus_hist_sum += color_hist_sum
    else:
        print(f"Failed to load image {img_path}")


# Process Others images
others_images = glob.glob(f'{data_set}/others/*.jpg')
for img_path in tqdm(others_images,desc="Processing Others Images"):
    
    img = cv.imread(img_path)
    
    if img is not None:
        color_hist_sum = compute_hsv_histogram(img, bins=8)
        others_data.append({
            "image_path": img_path,
            **{f"color_hist_{i}": color_hist_sum[i] for i in range(len(color_hist_sum))},
            "label": 1  # Others class label
        })
        others_hist_sum += color_hist_sum
    else:
        print(f"Failed to load image {img_path}")


if data_set == 'test':
    test_data = []
    test_hist_sum = np.zeros(24) 
    
    test_images = glob.glob(f'{data_set}/*.jpg')
    
    for img_path in tqdm(test_images, desc="Processing Test Images"):
        
        img = cv.imread(img_path)
        
        if img is not None:
            color_hist_sum = compute_hsv_histogram(img, bins=8)
            nevus_data.append({
                "image_path": img_path,
                **{f"color_hist_{i}": color_hist_sum[i] for i in range(len(color_hist_sum))},
            })
            test_hist_sum += color_hist_sum
        else:
            print(f"Failed to load image {img_path}")

    test_hist_sum /= len(test_images)

# Combine data from both classes and save to CSV
all_data = nevus_data + others_data + test_data
df = pd.DataFrame(all_data)
df.to_csv(save_full_path, index=False)
print(f"HSV color features saved to {save_full_path}")