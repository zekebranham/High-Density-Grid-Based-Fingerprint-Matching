from Lib.Probe_Processor import probe_processor, probe_dataBuilder
from Lib.high_density_matching import match_probe_fingerprint
import sqlite3
import time
import sys

def main():
    start_time_total = time.time()
    db_path = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project\DATABASE.db"
    probe_path = r"C:\Users\14807\Desktop\CSCI 158 Project\PROBE\008_L0_1.bmp"

    start_time_probe = time.time()
    probe_tripletslist = probe_processor(probe_path)
    probe_data = probe_dataBuilder(probe_tripletslist)
    #converts probe_data to strings
    probe_data = {str(key): value for key, value in probe_data.items()}
    end_time_probe = time.time()
    elapsed_time_probe = end_time_probe - start_time_probe

    start_time_db = time.time()
    try:
        conn = sqlite3.connect(db_path)
        print("Connected to the database successfully.")
    except sqlite3.Error as e:
        print(f"Failed to connect to the database: {e}")
        return
    end_time_db = time.time()
    elapsed_time_db = end_time_db - start_time_db

    # Call the matching function
    start_time_matching = time.time()
    best_match, best_score= match_probe_fingerprint(probe_data, db_path)
    end_time_matching = time.time()
    elapsed_time_matching = end_time_matching - start_time_matching

    print(f"The Probe Processing computation took {elapsed_time_probe} seconds.")
    print(f"The Database Processing computation took {elapsed_time_db} seconds.")
    print(f"The Matching computation took {elapsed_time_matching} seconds.")
    conn.close()
    # Output the result
    if best_match is not None:
        print(f"The best matching fingerprint ID is {best_match} with a score of {best_score:.2f}.")
    else:
        print(f"No matching fingerprint was found.")

    # Close the database connection
    conn.close()

    end_time_total = time.time()
    elapsed_time_total = end_time_total - start_time_total
    print(f"The Total High-Density Grid Matching Processing computation took {elapsed_time_total} seconds.")
if __name__ == '__main__':
    # Open a file to capture the output
    with open('output.txt', 'w') as f:
    # Redirect standard output to the file
        sys.stdout = f
        main()
    sys.stdout = sys.__stdout__