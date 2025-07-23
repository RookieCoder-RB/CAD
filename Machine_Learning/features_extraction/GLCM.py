import numpy as np
import pandas as pd
import cv2 as cv
import os
from skimage.feature import graycomatrix, graycoprops
import glob
from tqdm import tqdm

'''
Paramenters for GLCM

Distance(d): the displacement between two pixels

Angle(theta): the direction in which pixle pairs are considered

# of Gray Levels(G): The number of discrete intensity levels in the image.
'''

data_set = 'test'

save_path = f'results/{data_set}'
save_filename = f'glcm_features_{data_set}.csv'
save_full_path = os.path.join(save_path, save_filename)


os.makedirs(save_path, exist_ok=True)

# Parameters for GLCM
d = [50]
theta = [0, np.pi]  # 0 and 180 degrees

# Lists to hold features for both classes
data = []

# Process Nevus images
nevus_images = glob.glob(f'{data_set}/nevus/*.jpg')
for img in tqdm(nevus_images, desc="Processing Nevus Images"):
    img_glcm = cv.imread(img, cv.IMREAD_GRAYSCALE)
    if img_glcm is not None:
        glcm = graycomatrix(img_glcm, distances=d, angles=theta, levels=256)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        # homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        # ASM = graycoprops(glcm,'ASM').flatten() # uniformity
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        data.append({
            "image_path": img,
            "contrast_0": contrast[0],
            "contrast_180": contrast[1],
            "dissimilarity_0": dissimilarity[0],
            "dissimilarity_180": dissimilarity[1],
            # "homogeneity_0": homogeneity[0],
            # "homogeneity_180": homogeneity[1],
            # "ASM_0":ASM[0],
            # "ASM_180":ASM[1],
            "energy_0": energy[0],
            "energy_180": energy[1],
            "correlation_0": correlation[0],
            "correlation_180": correlation[1],
            "label": 0  # Nevus class label
        })

# Process Others images
others_images = glob.glob(f'{data_set}/others/*.jpg')

for img in tqdm(others_images, desc="Processing Others Images"):
    img_glcm = cv.imread(img, cv.IMREAD_GRAYSCALE)
    if img_glcm is not None:
        glcm = graycomatrix(img_glcm, distances=d, angles=theta, levels=256)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        # homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        # ASM = graycoprops(glcm,'ASM').flatten() # uniformity
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        data.append({
            "image_path": img,
            "contrast_0": contrast[0],
            "contrast_180": contrast[1],
            "dissimilarity_0": dissimilarity[0],
            "dissimilarity_180": dissimilarity[1],
            # "homogeneity_0": homogeneity[0],
            # "homogeneity_180": homogeneity[1],
            # "ASM_0":ASM[0], 
            # "ASM_180":ASM[1],
            "energy_0": energy[0],
            "energy_180": energy[1],
            "correlation_0": correlation[0],
            "correlation_180": correlation[1],
            "label": 1  # Others class label
        })

# Process Test images
if data_set == 'test':
    
    test_images = glob.glob(f'{data_set}/*.jpg')
    
    for img in tqdm(test_images, desc="Processing Test Images"):
        img_glcm = cv.imread(img, cv.IMREAD_GRAYSCALE)
        if img_glcm is not None:
            glcm = graycomatrix(img_glcm, distances=d, angles=theta, levels=256)
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            # homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            # ASM = graycoprops(glcm,'ASM').flatten() # uniformity
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()
            data.append({
                "image_path": img,
                "contrast_0": contrast[0],
                "contrast_180": contrast[1],
                "dissimilarity_0": dissimilarity[0],
                "dissimilarity_180": dissimilarity[1],
                # "homogeneity_0": homogeneity[0],
                # "homogeneity_180": homogeneity[1],
                # "ASM_0":ASM[0], 
                # "ASM_180":ASM[1],
                "energy_0": energy[0],
                "energy_180": energy[1],
                "correlation_0": correlation[0],
                "correlation_180": correlation[1],
            })

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to the specified path
df.to_csv(save_full_path, index=False)

print(f"Features saved to {save_full_path}")



