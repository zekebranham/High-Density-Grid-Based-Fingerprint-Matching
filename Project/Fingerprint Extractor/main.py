from Cascade1Fingerprint_Processor import process_fingerprint
from Cascade1Fingerprint_Processor import visualize_minutiae
from Cascade1Fingerprint_Processor import center_fingerprint_roi
from Cascade1Fingerprint_Processor import divide_into_grids
from Cascade1Fingerprint_Processor import save_cascade1_output
from database_worker import connect_db, create_tables, insert_data
from Fingerprint_Extractor.crossing_number import calculate_minutiaes
from Fingerprint_Extractor.crossing_number import remove_false_minutiae
from Cascade2Fingerprint_Processor import process_cascade2, MinutiaPoint
import os
import time
import json

#zeke

def process_and_store_images(folder_path, conn):
    create_tables(conn)

    for filename in os.listdir(folder_path):
        if filename.endswith(".bmp"):  # Adjust for your image file extensions
            image_path = os.path.join(folder_path, filename)
            output_dir = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project\Image Folder"  # Define or adjust the output directory for processed images
            try:
                # Insert a new fingerprint entry and get its ID
                cursor = conn.cursor()
                conn.execute('INSERT INTO Fingerprints (AdditionalInfo) VALUES (?)', ('Processed fingerprint',))
                fingerprint_id = cursor.lastrowid
            
                # Process the fingerprint image and get triplets data
                triplets_data = process_fingerprint_image(image_path, fingerprint_id)
                #print(f"Generated {len(triplets_data)} triplets for {filename}")  # Debugging statement
            
                if triplets_data:
                    # Assuming fingerprint_id is now available
                    insert_data(conn, fingerprint_id, triplets_data)

                conn.commit()
            except Exception as e:
                print(f"Failed to process {filename} due to error: {e}")
                conn.rollback()  # Rollback in case of error
    conn.close()

def process_fingerprint_image(image_path, fingerprint_id):
    # Load the image and perform initial processing
    img, mask = process_fingerprint(image_path)
    centered_img = center_fingerprint_roi(img, mask)
    
    if centered_img is not None:
        # Process image to extract minutiae
        minutiae_img, term_positions, bif_positions = calculate_minutiaes(centered_img, mask, kernel_size=3, edge_margin=10)
        true_term_positions, true_bif_positions = remove_false_minutiae(centered_img, term_positions, bif_positions, mask)
        
        # Divide into grids and identify high-density grids
        average_minutiae_per_grid, high_density_grid_ids, grid_size_x, grid_size_y = divide_into_grids(centered_img, mask, true_term_positions, true_bif_positions, 80, 80, 1.5)
        save_cascade1_output(high_density_grid_ids, true_term_positions, true_bif_positions)

        # Process the high-density grids to extract triplet data
        triplets_data = process_cascade2(grid_size_x, grid_size_y, centered_img.shape[1])
        
        return triplets_data
    else:
        print("Error: Image not found or failed to process.")
        return {}

if __name__ == "__main__":
    # Path to the fingerprint image file
    output_dir = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project" #use whatever path you want to view the image in, I suggest within the same directory as this file
    image_path = r"C:\Users\14807\Desktop\CSCI 158 Project\004\L\004_L2_4.bmp" #image path to process
    gridOutput_path = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project\GridOutputImages" #for filtered minutiae method of division
    folder_path = r"C:\Users\14807\Desktop\CSCI 158 Project\001\L" #for folder holding fingerprint images

    conn = connect_db()

    #time the program
    start_time = time.time()

    process_and_store_images(folder_path, conn)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"The computation took {elapsed_time} seconds.")
    conn.close()

    
        
