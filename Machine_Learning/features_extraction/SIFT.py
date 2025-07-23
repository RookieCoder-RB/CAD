import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

data_set = 'test'

save_path = f'results/{data_set}'
save_filename = f'sift_features_{data_set}.csv'
save_full_path = os.path.join(save_path, save_filename)

os.makedirs(save_path, exist_ok=True)

# Initialize SIFT
sift = cv.SIFT_create()

nevus_descriptors = []
others_descriptors = []

# Function to extract SIFT descriptors
def extract_sift_descriptors(image):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# Process Nevus images
nevus_images = glob.glob(f'{data_set}/nevus/*.jpg')
for img_path in tqdm(nevus_images, desc="Processing Nevus Images"):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    if img is not None:
        descriptors = extract_sift_descriptors(img)
        if descriptors is not None:
            nevus_descriptors.append(descriptors)
    else:
        print(f"Failed to load image {img_path}")

# Process Others images
others_images = glob.glob(f'{data_set}/others/*.jpg')
for img_path in tqdm(others_images,desc="Processing Others Images"):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    if img is not None:
        descriptors = extract_sift_descriptors(img)
        if descriptors is not None:
            others_descriptors.append(descriptors)
    else:
        print(f"Failed to load image {img_path}")
        
        
print("SIFT descriptors extracted.")

# Number of clusters (visual words)
n_clusters = 50

# Combine all descriptors for clustering
all_descriptors = np.vstack(nevus_descriptors + others_descriptors)

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(all_descriptors)

# Function to compute histogram of visual words
def compute_histogram(descriptors, kmeans, n_clusters):
    words, _ = vq(descriptors, kmeans.cluster_centers_)
    hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
    return hist / len(descriptors)  

# Lists to hold data for each class
data = []

# Process Nevus images to create fixed-length feature vectors
for idx, descriptors in enumerate(nevus_descriptors):
    hist = compute_histogram(descriptors, kmeans, n_clusters)
    data.append({
        "image_path": nevus_images[idx],
        **{f"visual_word_{i}": hist[i] for i in range(len(hist))},
        "label": 0  # Nevus class label
    })

# Process Others images to create fixed-length feature vectors
for idx, descriptors in enumerate(others_descriptors):
    hist = compute_histogram(descriptors, kmeans, n_clusters)
    data.append({
        "image_path": others_images[idx],
        **{f"visual_word_{i}": hist[i] for i in range(len(hist))},
        "label": 1  # Others class label
    })

# Convert to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv(save_full_path, index=False)
print(f"SIFT features saved to {save_full_path}")