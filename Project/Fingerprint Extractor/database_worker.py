import sqlite3
import json
#this file takes the fingerprint triplet data, and stores it into a SQLite database with 3 tables. One table for the fingerprint ID's, one for the High Density ID's, and one for the actuall triplet data
#next step is optimizing this and just including high density grids in the database and being able to call these specific fingerprint ID's with SQL

# Connect to the SQLite database (or create it if it doesn't exist)
def connect_db():
    return sqlite3.connect('fingerprint_data.db')

def create_tables(conn):
    cursor = conn.cursor()
    # Create tables (execute only once to setup)
    cursor.execute('''CREATE TABLE IF NOT EXISTS Fingerprints (FingerprintID INTEGER PRIMARY KEY AUTOINCREMENT, AdditionalInfo TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS HighDensityGrids (GridID INTEGER PRIMARY KEY AUTOINCREMENT, FingerprintID INTEGER, GridPosition TEXT, FOREIGN KEY(FingerprintID) REFERENCES Fingerprints(FingerprintID))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Triplets (TripletID INTEGER PRIMARY KEY AUTOINCREMENT, GridID INTEGER, MinutiaePoints TEXT, Distances TEXT, Angles TEXT, FOREIGN KEY(GridID) REFERENCES HighDensityGrids(GridID))''')
    conn.commit()

# Function to insert data into the databas
def insert_data(conn, fingerprint_id, triplets_data):
    cursor = conn.cursor()
    # Insert fingerprint data
    cursor.execute('INSERT INTO Fingerprints (FingerprintID, AdditionalInfo) VALUES (?, ?)', (fingerprint_id, 'Sample Info'))
    #print("Test1")
    print(fingerprint_id)
    # Insert high-density grid data and triplets
    for grid_id, triplets in triplets_data.items():
        cursor.execute('INSERT INTO HighDensityGrids (FingerprintID, GridPosition) VALUES (?, ?)', (fingerprint_id, grid_id))
        grid_id = cursor.lastrowid  # Get the auto-incremented GridID
        #print("Test2")
        
        for triplet in triplets:
            # Assuming triplet is a dict with 'MinutiaePoints', 'Distances', 'Angles'
            cursor.execute('INSERT INTO Triplets (GridID, MinutiaePoints, Distances, Angles) VALUES (?, ?, ?, ?)', (grid_id, json.dumps(triplet['MinutiaePoints']), json.dumps(triplet['Distances']), json.dumps(triplet['Angles'])))
            #print("Test3")

    conn.commit()


