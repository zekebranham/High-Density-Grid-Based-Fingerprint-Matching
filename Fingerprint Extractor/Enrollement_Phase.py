from Lib.database_worker import connect_db
from Lib.Database_Load_Processor import process_and_store_images
import time

#zeke

if __name__ == "__main__":
    #change to directory of your folder of fingerprint images to be enrolled
    folder_path = r"C:\Users\14807\Desktop\BIT Fingerprint Database\DATABASE"
    
    #Loading the Database
    conn = connect_db()

    start_time = time.time()
    fingerprint_id = process_and_store_images(folder_path, conn)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"The Database Loading Program computation took {elapsed_time} seconds.")
    conn.close()

    
        