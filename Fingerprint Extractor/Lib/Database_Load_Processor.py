
from Lib.database_worker import create_tables, insert_data, insert_fingerprint, get_next_fingerprint_id
from Fingerprint_Extractor.ExtractionONE import process_fingerprint
from Fingerprint_Extractor.ExtractionONE import visualize_minutiae
from Fingerprint_Extractor.ExtractionONE import divide_into_grids
from Fingerprint_Extractor.ExtractionONE import save_ExtractionONE_output
from Fingerprint_Extractor.Line_Scan_MD import calculate_minutiaes
from Fingerprint_Extractor.Line_Scan_MD import remove_false_minutiae
from Fingerprint_Extractor.Triplet_Extraction import process_extraction2
import os

def process_and_store_images(folder_path, conn):
    create_tables(conn)
    for filename in os.listdir(folder_path):
        if filename.endswith(".bmp"):  # Adjust for your image file extensions
            image_path = os.path.join(folder_path, filename)
            try:
                fingerprint_id = get_next_fingerprint_id(conn)
                #insert new starting fingerprint
                insert_fingerprint(conn, fingerprint_id, "SomeIdentifier")  # Insert the fingerprint record first
                #start_timeProcess1 = time.time()
                triplets_data = process_fingerprint_image(image_path)
                #end_timeProcess1 = time.time()
                #elapsed_timeProcess1 = end_timeProcess1 - start_timeProcess1
                #print(f"The Process1 computation took {elapsed_timeProcess1} seconds.")

                #fingerprint_id = fingerprint_id + 1
                if triplets_data:
                    #start_timeDB = time.time()
                    loop1, loop2, loop3 = insert_data(conn, fingerprint_id, triplets_data)  # Now insert the related grid and minutiae data
                    #print(f"loop1={loop1}")
                    #print(f"loop2={loop2}")
                    #print(f"loop3={loop3}")
                    #end_timeDB = time.time()
                    #elapsed_timeDB = end_timeDB - start_timeDB
                    #print(f"The Database computation took {elapsed_timeDB} seconds.")

            except Exception as e:
                print(f"Failed to process {filename} due to error: {e}")
                conn.rollback()
    conn.close()
    #added to prevent output NOPE
    #print(f"Loop1 = {loop1}")
    #print(f"Loop2 = {loop2}")
    #print(f"Loop3 = {loop3}")


    return fingerprint_id


def process_fingerprint_image(image_path):
    # Load the image and perform initial processing
    #start_timeEnhance = time.time()
    skeleton_img, centered_img, mask = process_fingerprint(image_path)
    #centered_img = center_fingerprint_roi(img, mask)
    #end_timeEnhance = time.time()
    #elapsed_timeEnhance = end_timeEnhance - start_timeEnhance
    #print(f"The Ehnacement computation took {elapsed_timeEnhance} seconds.")

    
    if centered_img is not None:
        # Process image to extract minutiae
        #start_timeMinutiae1 = time.time()
        minutiae_img, term_positions, bif_positions = calculate_minutiaes(centered_img, mask, kernel_size=3, edge_margin=10)
        true_term_positions, true_bif_positions = remove_false_minutiae(centered_img, term_positions, bif_positions, mask)
        #end_timeMinutiae1 = time.time()
        #elapsed_timeMinutiae1 = end_timeMinutiae1 - start_timeMinutiae1
        #print(f"The Minutiae1 computation took {elapsed_timeMinutiae1} seconds.")

        #start_timeGrids = time.time()
        # Divide into grids and identify high-density grids
        average_minutiae_per_grid, high_density_grid_ids, grid_size_x, grid_size_y = divide_into_grids(centered_img, mask, true_term_positions, true_bif_positions, 80, 80)
        #end_timeGrids = time.time()
        #elapsed_timeGrids = end_timeGrids - start_timeGrids
        #print(f"The Grid Division computation took {elapsed_timeGrids} seconds.")
        save_ExtractionONE_output(high_density_grid_ids, true_term_positions, true_bif_positions)

        # Process the high-density grids to extract triplet data
        #start_timeTrip = time.time()
        triplets_data = process_extraction2(grid_size_x, grid_size_y, centered_img.shape[1])

        #end_timeTrip = time.time()
        #elapsed_timeTrip = end_timeTrip - start_timeTrip
        #print(f"The Triplets computation took {elapsed_timeTrip} seconds.")

        return triplets_data
    else:
        print("Error: Image not found or failed to process.")
        #return {}
