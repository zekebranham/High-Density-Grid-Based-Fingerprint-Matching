import cv2 as cv
import numpy
import math
from glob import glob
from Fingerprint_Extractor import feature_extractor
from image_enhancement.normalization import normalize
from image_enhancement.segmentation import create_segmented_and_variance_images
from image_enhancement.orientation import calculate_angles
from image_enhancement.frequency import ridge_freq
from image_enhancement.gabor_filter import gabor_filter
from image_enhancement.skeletonize import skeletonize
from image_enhancement.crossing_number import calculate_minutiaes

#zeke

def load_image(image_path):
    return cv.imread(image_path, cv.IMREAD_GRAYSCALE)

def process_fingerprint(image_path, output_dir):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Perform image enhancement and processing steps
    normalized_img = normalize(img, 100, 100)
    #cv.imwrite(f'{output_dir}/01_normalizedNEW2.jpg', normalized_img)
    segmented_img, norm_img, mask = create_segmented_and_variance_images(normalized_img, 16, 0.2)
    #cv.imwrite(f'{output_dir}/02_segmentedNEW2.jpg', segmented_img)
    angles = calculate_angles(norm_img, 16)
    freq = ridge_freq(norm_img, mask, angles, 16, 5, 5, 15)
    gabor_img = gabor_filter(norm_img, angles, freq)
    #cv.imwrite(f'{output_dir}/03_gabor_filteredNEW2.jpg', gabor_img)
    skeleton_img = skeletonize(gabor_img)
    cv.imwrite(f'{output_dir}/04_skeleton.jpg', skeleton_img)
    #minutiae_img = calculate_minutiaes(skeleton_img, kernel_size=3, edge_margin=2)

    return skeleton_img
    #return minutiae_img

def minutiae_counts(img, spuriousMinutiaeThresh=10, invertImage=False):
    extractor = feature_extractor.SimplifiedFingerprintFeatureExtractor()
    #extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if invertImage:
        img = 255 - img
    #DELETE minutiaeTermPositions, minutiaeBifPositions
    minutiaeTermCount, minutiaeBifCount, minutiaeTermPositions, minutiaeBifPositions = extractor.getTerminationBifurcation(img)
    #DELETE minutiaeTermPosition, minutiaeBifPosition
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
    for (x, y) in term_positions:
        cv.circle(vis_image, (x, y), 2, (0, 0, 255), -1)  # Red for terminations
    # Draw bifurcation minutiae in green
    for (x, y) in bif_positions:
        cv.circle(vis_image, (x, y), 2, (0, 255, 0), -1)  # Green for bifurcations

    cv.imwrite(f'{output_dir}/minutiaeVisualisation2.jpg', vis_image)  # Save the visualized image


if __name__ == "__main__":
    # Path to the fingerprint image file
    output_dir = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project" #use whatever path you want to view the image in, I suggest within the same directory as this file
    output_dir2 = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project" #You dont really need this one, but just for path clarity
    image_path = r"C:\Users\14807\Desktop\CSCI 158 Project\022\L\022_L1_0.bmp" #image path to process
    skel_image_path = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project/04_skeleton.jpg'" #dont really need
    
    # Load the image
    img = process_fingerprint(image_path, output_dir)
    height, width = img.shape
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found.")
    else:
        
        #Process image and get minutiae visualization and positions
        minutiae_img, term_positions, bif_positions = calculate_minutiaes(img, kernel_size=3, edge_margin=2)
        
        # Divide the image into grids and store minutiae counts and positions for each grid
        #grid_minutiae_counts = {}  # Stores the count of minutiae for each grid
        #grid_minutiae_positions = {}  # Stores the list of minutiae positions for each grid
        grid_terminations = {}
        grid_bifurcations = {}
        grid_minutiae_counts = {}  # Initialize the dictionary to store the count of minutiae per grid
        # Print the counts
        
        grid_size_x = 80  # Example grid size, adjust as needed
        grid_size_y = 80  # Example grid size, adjust as needed
        grid_count_x = math.ceil(width / grid_size_x)  # Calculate number of horizontal grids
        grid_count_y = math.ceil(height / grid_size_y)  # Calculate number of vertical grids
        grid_count = grid_count_x * grid_count_y  # Total grid count
        # Count terminations and bifurcations for each grid
        for term in term_positions:
            x, y = term
            grid_x = x // grid_size_x
            grid_y = y // grid_size_y
            grid_id = grid_y * (width // grid_size_x) + grid_x #set the ceiling for total number of grids
            grid_terminations[grid_id] = grid_terminations.get(grid_id, 0) + 1 #increment the termination list
            grid_minutiae_counts[grid_id] = grid_minutiae_counts.get(grid_id, 0) + 1 #increment the total minutiae list

        for bif in bif_positions:
            x, y = bif
            grid_x = x // grid_size_x
            grid_y = y // grid_size_y
            grid_id = grid_y * (width // grid_size_x) + grid_x #set ceiling for total number of grids
            grid_bifurcations[grid_id] = grid_bifurcations.get(grid_id, 0) + 1 #increment the bifucation list
            grid_minutiae_counts[grid_id] = grid_minutiae_counts.get(grid_id, 0) + 1 #increment total minutiae list

        # Print the detailed counts for each grid
        for grid_id in range(grid_count):
            num_terminations = grid_terminations.get(grid_id, 0)
            num_bifurcations = grid_bifurcations.get(grid_id, 0)
            print(f"Grid {grid_id}: Terminations = {num_terminations}, Bifurcations = {num_bifurcations}")

        print(img.shape)

        # Calculate the average number of minutiae per grid
        total_minutiae = sum(grid_minutiae_counts.values())
        average_minutiae_per_grid = total_minutiae / grid_count

        # Define a threshold for high-density grids
        # For example, select grids with at least 20% more minutiae than the average
        threshold = average_minutiae_per_grid * 1.2

        # List to store the IDs of high-density grids
        high_density_grid_ids = []

        # Iterate through the grid_minutiae_counts and select high-density grids
        for grid_id, minutiae_count in grid_minutiae_counts.items():
            if minutiae_count > threshold:
                high_density_grid_ids.append(grid_id)

        # Now you have a list of grid IDs that have minutiae counts above the threshold
        print("High-density grid IDs:", high_density_grid_ids)

        # After obtaining minutiae positions...
        visualize_minutiae(img, term_positions, bif_positions, output_dir2)
