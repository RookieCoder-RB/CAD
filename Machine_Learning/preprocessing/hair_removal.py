import cv2
import numpy as np
import os
import glob
from tqdm import tqdm  # Progress bar for tracking

def dull_razor(img, kernel_size=7, inpaint_radius=3, resize_factor=0.5):
    """
    Optimized Dull Razor algorithm for faster hair removal.
    """
    # Resize to reduce computation
    img_resized = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply black-hat filtering with a smaller kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create a mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the detected hair regions with a smaller radius
    inpainted_img = cv2.inpaint(img_resized, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    # Resize back to original if necessary
    inpainted_img = cv2.resize(inpainted_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    
    return inpainted_img

def process_images(data_set):
    # Define input and output paths
    input_base_path = os.path.join(data_set)
    output_base_path = os.path.join(f"{data_set}_HairRemoved")

    # Create output directories
    os.makedirs(os.path.join(output_base_path, 'nevus'), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'others'), exist_ok=True)

    # Process images in nevus and others folders
    nevus_images = glob.glob(os.path.join(input_base_path, 'nevus', '*.jpg'))
    others_images = glob.glob(os.path.join(input_base_path, 'others', '*.jpg'))

    # Process nevus images with progress bar
    for img_path in tqdm(nevus_images, desc="Processing Nevus Images"):
        img = cv2.imread(img_path)
        if img is not None:
            result = dull_razor(img)
            img_name = os.path.basename(img_path)
            save_path = os.path.join(output_base_path, 'nevus', img_name)
            cv2.imwrite(save_path, result)
    
    # Process others images with progress bar
    for img_path in tqdm(others_images, desc="Processing Others Images"):
        img = cv2.imread(img_path)
        if img is not None:
            result = dull_razor(img)
            img_name = os.path.basename(img_path)
            save_path = os.path.join(output_base_path, 'others', img_name)
            cv2.imwrite(save_path, result)

    print("Hair removal processing complete for all images.")

# Run the optimized process
process_images('val')
