import math
from glob import glob
from Cascade1Fingerprint_Processor import process_fingerprint, visualize_minutiae, center_fingerprint_roi, save_cascade1_output
from image_enhancement.crossing_number import calculate_minutiaes
from Cascade2Fingerprint_Processor import process_cascade2, MinutiaPoint

#zeke

if __name__ == "__main__":
    # Path to the fingerprint image file
    output_dir = r"C:\Users\jonat\Documents\miniconda\High-Density-Grid-Based-Fingerprint-Matching\Project" 
#use whatever path you want to view the image in, I suggest within the same directory as this file
    output_dir2 = r"C:\Users\jonat\Documents\miniconda\High-Density-Grid-Based-Fingerprint-Matching\Project"
#You dont really need this one, but just for path clarity
    image_path = r"C:\Users\jonat\Documents\miniconda\High-Density-Grid-Based-Fingerprint-Matching\database\002\R\002_R3_3.bmp"
#image path to process
    skel_image_path = r"C:\Users\jonat\Documents\miniconda\High-Density-Grid-Based-Fingerprint-Matching\Project\04_skeleton.jpg" #dont really need

    image2_path = r"C:\Users\jonat\Documents\miniconda\High-Density-Grid-Based-Fingerprint-Matching\database\002\R\002_R3_3.bmp"

    #firstFinger
    # Load the image
    img, mask = process_fingerprint(image_path, output_dir)
    # Center the ROI of the fingerprint in the image
    centered_img = center_fingerprint_roi(img, mask)
    height, width = centered_img.shape
    #height, width = img.shape

    # Check if the image was loaded successfully
    if centered_img is None:
    #if img is None:
        print("Error: Image not found.")
    else:
        
        #Process image and get minutiae visualization and positions
        minutiae_img, raw_term_positions, raw_bif_positions = calculate_minutiaes(centered_img, mask, kernel_size=3, edge_margin=20)

        #minutiae_img, term_positions, bif_positions = calculate_minutiaes(img, mask, kernel_size=3, edge_margin=20)
        term_positions = [MinutiaPoint(x, y) for x, y in raw_term_positions]
        bif_positions = [MinutiaPoint(x, y) for x, y in raw_bif_positions]


        
        # Divide the image into grids and store minutiae counts and positions for each grid
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
            x, y = term.x, term.y
            grid_x = x // grid_size_x
            grid_y = y // grid_size_y
            grid_id = grid_y * (width // grid_size_x) + grid_x #set the ceiling for total number of grids
            grid_terminations[grid_id] = grid_terminations.get(grid_id, 0) + 1 #increment the termination list
            grid_minutiae_counts[grid_id] = grid_minutiae_counts.get(grid_id, 0) + 1 #increment the total minutiae list

        for bif in bif_positions:
            x, y = bif.x, bif.y
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

        print(centered_img.shape)        
        #print(img.shape)


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
        print("FP1 High-density grid IDs:", high_density_grid_ids)
        
        save_cascade1_output(high_density_grid_ids, term_positions, bif_positions)



# Now that cascade1_output.json should exist, process with Cascade 2
        highest_density_grid_id = max(grid_minutiae_counts, key=grid_minutiae_counts.get)
        process_cascade2(highest_density_grid_id, grid_size_x, grid_size_y, width)


# After obtaining minutiae positions, visualize them
        visualize_minutiae(img, term_positions, bif_positions, output_dir2)



"""
        #secondFinger
        # Load the image
    img2, mask2 = process_fingerprint(image2_path, output_dir)
    # Center the ROI of the fingerprint in the image
    centered_img2 = center_fingerprint_roi(img2, mask)
    height2, width2 = centered_img2.shape    
    #height2, width2 = img2.shape

    # Check if the image was loaded successfully
    if centered_img2 is None:   
    #if img2 is None:

        print("Error: Image not found.")
    else:
        
        #Process image and get minutiae visualization and positions
        minutiae_img2, term_positions2, bif_positions2 = calculate_minutiaes(centered_img2, mask2, kernel_size=3, edge_margin=20)
        #minutiae_img2, term_positions2, bif_positions2 = calculate_minutiaes(img2, mask2, kernel_size=3, edge_margin=20)
        
        # Divide the image into grids and store minutiae counts and positions for each grid
        grid_terminations2 = {}
        grid_bifurcations2 = {}
        grid_minutiae_counts2 = {}  # Initialize the dictionary to store the count of minutiae per grid
        # Print the counts
        
        grid_size_x2 = 80  # Example grid size, adjust as needed
        grid_size_y2 = 80  # Example grid size, adjust as needed
        grid_count_x2 = math.ceil(width2 / grid_size_x2)  # Calculate number of horizontal grids
        grid_count_y2 = math.ceil(height2 / grid_size_y2)  # Calculate number of vertical grids
        grid_count2 = grid_count_x2 * grid_count_y2  # Total grid count
        # Count terminations and bifurcations for each grid
        for term2 in term_positions2:
            x2, y2 = term2
            grid_x2 = x2 // grid_size_x2
            grid_y2 = y2 // grid_size_y2
            grid_id2 = grid_y2 * (width2 // grid_size_x2) + grid_x2 #set the ceiling for total number of grids
            grid_terminations2[grid_id2] = grid_terminations2.get(grid_id2, 0) + 1 #increment the termination list
            grid_minutiae_counts2[grid_id2] = grid_minutiae_counts2.get(grid_id2, 0) + 1 #increment the total minutiae list

        for bif2 in bif_positions2:
            x2, y2 = bif2
            grid_x2 = x2 // grid_size_x2
            grid_y2 = y2 // grid_size_y2
            grid_id2 = grid_y2 * (width2 // grid_size_x2) + grid_x2 #set ceiling for total number of grids
            grid_bifurcations2[grid_id2] = grid_bifurcations2.get(grid_id2, 0) + 1 #increment the bifucation list
            grid_minutiae_counts2[grid_id2] = grid_minutiae_counts2.get(grid_id2, 0) + 1 #increment total minutiae list

        # Print the detailed counts for each grid
        for grid_id2 in range(grid_count2):
            num_terminations2 = grid_terminations2.get(grid_id2, 0)
            num_bifurcations2 = grid_bifurcations2.get(grid_id2, 0)
            print(f"Grid {grid_id2}: Terminations = {num_terminations2}, Bifurcations = {num_bifurcations2}")

        print(centered_img2.shape)        
        #print(img2.shape)


        # Calculate the average number of minutiae per grid
        total_minutiae2 = sum(grid_minutiae_counts2.values())
        average_minutiae_per_grid2 = total_minutiae2 / grid_count2

        # Define a threshold for high-density grids
        # For example, select grids with at least 20% more minutiae than the average
        threshold2 = average_minutiae_per_grid2 * 1.2

        # List to store the IDs of high-density grids
        high_density_grid_ids2 = []

        # Iterate through the grid_minutiae_counts and select high-density grids
        for grid_id2, minutiae_count2 in grid_minutiae_counts2.items():
            if minutiae_count2 > threshold2:
                high_density_grid_ids2.append(grid_id2)

        # Now you have a list of grid IDs that have minutiae counts above the threshold
        print("FP2 High-density grid IDs:", high_density_grid_ids2)

        # After obtaining minutiae positions...
        #visualize_minutiae(centered_img2, term_positions2, bif_positions2, output_dir2)
        visualize_minutiae(img2, term_positions2, bif_positions2, output_dir2)

"""
        
