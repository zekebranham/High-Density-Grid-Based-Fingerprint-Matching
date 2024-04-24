
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
                triplets_data = process_fingerprint_image(image_path)
                if triplets_data:
                    insert_data(conn, fingerprint_id, triplets_data)  # Now insert the related grid and minutiae data
            except Exception as e:
                print(f"Failed to process {filename} due to error: {e}")
                conn.rollback()
    conn.close()
    return fingerprint_id


def process_fingerprint_image(image_path):
    # Load the image and perform initial processing
    skeleton_img, centered_img, mask = process_fingerprint(image_path)
    
    if centered_img is not None:
        # Process image to extract minutiae
        minutiae_img, term_positions, bif_positions = calculate_minutiaes(centered_img, mask, kernel_size=3, edge_margin=10)
        true_term_positions, true_bif_positions = remove_false_minutiae(centered_img, term_positions, bif_positions, mask)

        # Divide into grids and identify high-density grids
        average_minutiae_per_grid, high_density_grid_ids, grid_size_x, grid_size_y = divide_into_grids(centered_img, mask, true_term_positions, true_bif_positions, 80, 80)
        save_ExtractionONE_output(high_density_grid_ids, true_term_positions, true_bif_positions)

        # Process the high-density grids to extract triplet data
        triplets_data = process_extraction2(grid_size_x, grid_size_y, centered_img.shape[1])

        return triplets_data
    else:
        print("Error: Image not found or failed to process.")
