import cv2 as cv
import numpy as np
import math
import os
import json
from image_enhancement.normalization import normalize
from image_enhancement.segmentation import create_segmented_and_variance_images
from image_enhancement.orientation import calculate_angles
from image_enhancement.frequency import ridge_freq
from image_enhancement.gabor_filter import gabor_filter
from image_enhancement.skeletonize import skeletonize

#zeke
""""
This file is souly for image enhancement and preperation. By taking the input image, using normalization, 
segmentation, ridge frequency detection, gabor filtering, skeletonization, image centering, surface
minutiae detection, and image visualization, we prepare the image to be processed at the next cascade

"""
def load_image(image_path):
    return cv.imread(image_path, cv.IMREAD_GRAYSCALE)

def process_fingerprint(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Perform image enhancement and processing steps
    normalized_img = normalize(img, 100, 100)
    #cv.imwrite(f'{output_dir}/01_normalized.jpg', normalized_img)
    segmented_img, norm_img, mask = create_segmented_and_variance_images(normalized_img, 16, 0.2)  #set the background threshhold to 0.37 and window size to 8
    #cv.imwrite(f'{output_dir}/02_segmented.jpg', segmented_img)
    angles = calculate_angles(norm_img, 16)
    freq = ridge_freq(norm_img, mask, angles, 16, 5, 5, 15)
    gabor_img = gabor_filter(norm_img, angles, freq)
    #cv.imwrite(f'{output_dir}/03_gabor_filtered.jpg', gabor_img)
    skeleton_img = skeletonize(gabor_img)
    #cv.imwrite(f'{output_dir}/04_skeleton.jpg', skeleton_img)
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

def adjust_grid_size(image_dim, desired_grid_dim):
    """
    Adjusts the grid size slightly to ensure it fits evenly into the image dimensions.
    
    Args:
        image_dim (int): One dimension (height or width) of the image.
        desired_grid_dim (int): The desired dimension (height or width) of each grid.
    
    Returns:
        int: The adjusted dimension of the grid that ensures an even fit across the image.
    """
    num_grids = math.ceil(image_dim / desired_grid_dim)
    adjusted_grid_dim = image_dim // num_grids
    return adjusted_grid_dim

#version to calculate average amount of minutiae per grid
#def divide_into_grids(image, mask, term_positions, bif_positions, output_dir, grid_size_x=80, grid_size_y=80, threshold=1.2):
def divide_into_grids(image, mask, term_positions, bif_positions, desired_grid_size_x=80, desired_grid_size_y=80, threshold=1.2):
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    
    height, width = image.shape[:2]
    grid_size_x = adjust_grid_size(width, desired_grid_size_x)
    grid_size_y = adjust_grid_size(height, desired_grid_size_y)
    grid_terminations = {}
    grid_bifurcations = {}
    grid_minutiae_counts = {}  # Initialize the dictionary to store the count of minutiae per grid

    #for x in range(0, width, grid_size_x):
        #for y in range(0, height, grid_size_y):
            #grid_image = image[y:y + grid_size_y, x:x + grid_size_x]
            #grid_id = (y // grid_size_y) * (width // grid_size_x) + (x // grid_size_x)
            #filename = f"grid_{grid_id}.jpg"
            #cv.imwrite(os.path.join(output_dir, filename), grid_image)

    for x in range(0, width, grid_size_x):
        for y in range(0, height, grid_size_y):
            # Compute the grid ID
            grid_id = (y // grid_size_y) * (math.ceil(width / grid_size_x)) + (x // grid_size_x)
            #grid_image = image[y:y + grid_size_y, x:x + grid_size_x] #needed for output image

            #save each grid as an image
            #grid_filename = os.path.join(output_dir, f"grid_{grid_id}.jpg")
            #cv.imwrite(grid_filename, grid_image)

            # Filter termination and bifurcation points that fall within the current grid
            filtered_terms = [(tx, ty) for tx, ty in term_positions if x <= tx < x + grid_size_x and y <= ty < y + grid_size_y]
            filtered_bifs = [(bx, by) for bx, by in bif_positions if x <= bx < x + grid_size_x and y <= by < y + grid_size_y]
            
            # Count the minutiae in the grid
            grid_terminations[grid_id] = len(filtered_terms)
            grid_bifurcations[grid_id] = len(filtered_bifs)
            grid_minutiae_counts[grid_id] = len(filtered_terms) + len(filtered_bifs)
                # Print the detailed counts for each grid
    
    # Calculate average minutiae per grid and identify high-density grids
    total_minutiae = sum(grid_minutiae_counts.values())
    grid_count = len(grid_minutiae_counts)
    for grid_id in range(grid_count):
        num_terminations = grid_terminations.get(grid_id, 0)
        num_bifurcations = grid_bifurcations.get(grid_id, 0)
        #print(f"Grid {grid_id}: Terminations = {num_terminations}, Bifurcations = {num_bifurcations}")

    average_minutiae_per_grid = total_minutiae / grid_count if grid_count else 0
    high_density_grid_ids = [grid_id for grid_id, count in grid_minutiae_counts.items() if count > average_minutiae_per_grid * threshold]

    return average_minutiae_per_grid, high_density_grid_ids, grid_size_x, grid_size_y
    

def visualize_minutiae(img, term_positions, bif_positions, output_dir):
    vis_image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Convert to BGR for coloring
    # Draw termination minutiae in red
    for (x, y) in term_positions:
        cv.circle(vis_image, (x, y), 2, (0, 0, 255), -1)  # Red for terminations
    # Draw bifurcation minutiae in green
    for (x, y) in bif_positions:
        cv.circle(vis_image, (x, y), 2, (0, 255, 0), -1)  # Green for bifurcations

    cv.imwrite(f'{output_dir}/linescanWindow3.jpg', vis_image)  # Save the visualized image

    #Saves data used for cascade 2
def save_cascade1_output(high_density_grid_ids, term_positions, bif_positions):
    data_to_save = {
        'high_density_grid_ids': high_density_grid_ids,
        'minutiae': {
            'terminations': [(pt[0], pt[1]) for pt in term_positions],
            'bifurcations': [(pt[0], pt[1]) for pt in bif_positions]
        }
    }

    with open('cascade1_output.json', 'w') as f:
        json.dump(data_to_save, f)
