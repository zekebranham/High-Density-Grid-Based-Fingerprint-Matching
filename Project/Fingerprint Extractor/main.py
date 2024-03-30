from Cascade1Fingerprint_Processor import process_fingerprint
from Cascade1Fingerprint_Processor import visualize_minutiae
from Cascade1Fingerprint_Processor import center_fingerprint_roi
from Cascade1Fingerprint_Processor import divide_into_grids
from Fingerprint_Extractor.crossing_number import calculate_minutiaes
from Fingerprint_Extractor.crossing_number import remove_false_minutiae
#zeke

if __name__ == "__main__":
    # Path to the fingerprint image file
    output_dir = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project" #use whatever path you want to view the image in, I suggest within the same directory as this file
    output_dir2 = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project" #You dont really need this one, but just for path clarity
    image_path = r"C:\Users\14807\Desktop\CSCI 158 Project\004\L\004_L2_4.bmp" #image path to process
    skel_image_path = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project/04_skeleton.jpg'" #dont really need
    gridOutput_path = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project\GridOutputImages" #for filtered minutiae method of division
    gridOutput_path2 = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project\GridOutputImages2" #for traditional method of division
    gridOutput_path3 = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project\GridOutputImages3" #attempt at uniform grids

    image2_path = r"C:\Users\14807\Desktop\CSCI 158 Project\007\L\007_L0_4.bmp"

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
        minutiae_img, term_positions, bif_positions = calculate_minutiaes(centered_img, mask, kernel_size=3, edge_margin=10)
        #minutiae_img, term_positions, bif_positions = calculate_minutiaes(img, mask, kernel_size=3, edge_margin=20)

        #Remove false minutiae
        true_term_positions, true_bif_positions = remove_false_minutiae(centered_img, term_positions, bif_positions, mask)
        
        #Dividing into grids test
        average_minutiae_per_grid, high_density_grid_ids = divide_into_grids(centered_img, mask, true_term_positions, true_bif_positions, gridOutput_path3, 80, 80, 1.5) 
        print(f"average minutiae per grid: {average_minutiae_per_grid}")
        print(f"FP1 High-Density Grid ID's: {high_density_grid_ids}")
        print(centered_img.shape)  

        # After obtaining minutiae positions...
        visualize_minutiae(centered_img, true_term_positions, true_bif_positions, output_dir2)
        #visualize_minutiae(img, term_positions, bif_positions, output_dir2)
        
    
        