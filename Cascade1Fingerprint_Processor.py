import cv2 as cv
import numpy as np
import json
from glob import glob
from Fingerprint_Extractor import feature_extractor
from image_enhancement.normalization import normalize
from image_enhancement.segmentation import create_segmented_and_variance_images
from image_enhancement.orientation import calculate_angles
from image_enhancement.frequency import ridge_freq
from image_enhancement.gabor_filter import gabor_filter
from image_enhancement.skeletonize import skeletonize

#zeke

def load_image(image_path):
    return cv.imread(image_path, cv.IMREAD_GRAYSCALE)

def process_fingerprint(image_path, output_dir):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Perform image enhancement and processing steps
    normalized_img = normalize(img, 100, 100)
    cv.imwrite(f'{output_dir}/01_normalized.jpg', normalized_img)
    segmented_img, norm_img, mask = create_segmented_and_variance_images(normalized_img, 16, 0.2)  #set the background threshhold to 0.37 and window size to 8
    cv.imwrite(f'{output_dir}/02_segmented.jpg', segmented_img)
    angles = calculate_angles(norm_img, 16)
    freq = ridge_freq(norm_img, mask, angles, 16, 5, 5, 15)
    gabor_img = gabor_filter(norm_img, angles, freq)
    cv.imwrite(f'{output_dir}/03_gabor_filtered.jpg', gabor_img)
    skeleton_img = skeletonize(gabor_img)
    cv.imwrite(f'{output_dir}/04_skeleton.jpg', skeleton_img)
    #minutiae_img = calculate_minutiaes(skeleton_img, kernel_size=3, edge_margin=2)

    return skeleton_img, mask
    #return minutiae_img

def center_fingerprint_roi(img, mask):
    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour corresponds to the fingerprint
    if contours:
        cnt = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(cnt)  # Get the bounding box of the fingerprint
        
        # Calculate the center of the bounding box
        cx, cy = x + w // 2, y + h // 2
        
        # Calculate the center of the image
        img_cx, img_cy = img.shape[1] // 2, img.shape[0] // 2
        
        # Calculate the offsets needed to center the bounding box in the image
        offset_x, offset_y = img_cx - cx, img_cy - cy
        
        # Initialize a new image frame with the same size as the original image
        new_image = np.zeros_like(img)
        
        # Ensure the offsets do not exceed the original image's bounds
        # This is a simple check; a more robust solution may require additional checks
        if 0 <= y + offset_y < img.shape[0] and 0 <= x + offset_x < img.shape[1]:
            # Place the ROI in the center of the new image frame
            new_image[y + offset_y:y + offset_y + h, x + offset_x:x + offset_x + w] = img[y:y + h, x:x + w]
        
        return new_image

    return img  # Return the original if no contours found

def minutiae_counts(img, spuriousMinutiaeThresh=10, invertImage=False):
    extractor = feature_extractor.SimplifiedFingerprintFeatureExtractor()
    if invertImage:
        img = 255 - img
    minutiaeTermCount, minutiaeBifCount, minutiaeTermPositions, minutiaeBifPositions = extractor.getTerminationBifurcation(img)
    return minutiaeTermCount, minutiaeBifCount, minutiaeTermPositions, minutiaeBifPositions

#version to calculate average amount of minutiae per grid
def divide_into_grids(image, grid_size_x, grid_size_y, minutiaeTermPositions, minutiaeBifPositions):
    grids = []
    grid_minutiae_counts = {} #dictionary to store minutiae count for each grid
    total_minutiae = 0
    height, width = image.shape
    grid_count = 0
    grid_id = 0 #identifier
    for x in range(0, width, grid_size_x):
        for y in range(0, height, grid_size_y):
            grid = image[y:y+grid_size_y, x:x+grid_size_x]
            grids.append(grid)
            
            grid_count += 1

            
            grid_id += 1

    #calc avg minutiae count per grid
        if grid_count > 0:
            average_minutiae_per_grid = total_minutiae / grid_count
        else:
            average_minutiae_per_grid = 0
            print(f"Average Minutaie per grid: {average_minutiae_per_grid}")      
    return grids, average_minutiae_per_grid, total_minutiae, grid_count, grid_minutiae_counts

#DELETE
def visualize_minutiae(img, term_positions, bif_positions, output_dir):
    vis_image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Convert to BGR for coloring

    # Draw termination minutiae in red
    for term in term_positions:
        cv.circle(vis_image, (term.x, term.y), 2, (0, 0, 255), -1)  # Red for terminations

    # Draw bifurcation minutiae in green
    for bif in bif_positions:
        cv.circle(vis_image, (bif.x, bif.y), 2, (0, 255, 0), -1)  # Green for bifurcations

    cv.imwrite(f'{output_dir}/minutiaeVisualisation.jpg', vis_image)  # Save the visualized image

    
#Saves data used for cascade 2
def save_cascade1_output(high_density_grid_ids, term_positions, bif_positions):
    data_to_save = {
        'high_density_grid_ids': high_density_grid_ids,
        'minutiae': {
            'terminations': [(pt.x, pt.y) for pt in term_positions],
            'bifurcations': [(pt.x, pt.y) for pt in bif_positions]
        }
    }

    with open('cascade1_output.json', 'w') as f:
        json.dump(data_to_save, f)