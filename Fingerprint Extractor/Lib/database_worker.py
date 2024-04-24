import sqlite3
import json
#this file takes the fingerprint triplet data, and stores it into a SQLite database with 3 tables. One table for the fingerprint ID's, one for the High Density ID's, and one for the actuall triplet data
#next step is optimizing this and just including high density grids in the database and being able to call these specific fingerprint ID's with SQL

# Connect to the SQLite database (or create it if it doesn't exist)
def connect_db():
    return sqlite3.connect('DATABASE.db')

def create_tables(conn):
    cursor = conn.cursor()
    # Create tables (execute only once to setup)
    cursor.execute(
    '''CREATE TABLE Fingerprints (
    FingerprintID INTEGER PRIMARY KEY,
    PersonIdentifier TEXT)'''
    )
    cursor.execute(
    '''CREATE TABLE HighDensityGrids (
    GridID INTEGER PRIMARY KEY,
    FingerprintID INTEGER,
    GridLabel TEXT,
    FOREIGN KEY (FingerprintID) REFERENCES Fingerprints(FingerprintID));'''
    )
    cursor.execute(
    '''CREATE TABLE MinutiaeData (
    TripletID INTEGER PRIMARY KEY AUTOINCREMENT,
    FingerprintID INTEGER,
    GridLabel TEXT,
    MinutiaePoints TEXT,
    Distance TEXT,
    Angles TEXT,
    FOREIGN KEY (GridLabel) REFERENCES HighDensityGrids(GridID),
    FOREIGN KEY (FingerprintID) REFERENCES HighDensityGrids(FingerprintID));'''
    )
    conn.commit()

def insert_fingerprint(conn, fingerprint_id, person_identifier):
        cursor = conn.cursor()
        # Attempt to insert a new fingerprint, ensuring 'fingerprint_id' is unique
        cursor.execute("INSERT INTO Fingerprints (FingerprintID, PersonIdentifier) VALUES (?, ?)", (fingerprint_id, person_identifier))
        conn.commit()

def get_next_fingerprint_id(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(FingerprintID) FROM Fingerprints")
    last_id = cursor.fetchone()[0]
    if last_id is None:
        return 1  # Return 1 if the table is empty
    else:
        return last_id + 1
    
# Function to insert data into the databas
def insert_data(conn, fingerprint_id, triplets_data):
    loop1 = 0
    loop2 = 0
    loop3 = 0
    cursor = conn.cursor()
    for grid_label, data in triplets_data.items():
        loop1 = loop1 +1
        cursor.execute('INSERT INTO HighDensityGrids (FingerprintID, GridLabel) VALUES (?, ?)', (fingerprint_id, grid_label))
        
        for triplet in data['Triplets']:
            loop2 = loop2 +1
            minutiae_points = json.dumps(triplet['MinutiaePoints'])
            # Check if the triplet already exists
            cursor.execute('''SELECT EXISTS(SELECT 1 FROM MinutiaeData 
                             WHERE FingerprintID=? AND GridLabel=? AND MinutiaePoints=? LIMIT 1)''', 
                           (fingerprint_id, grid_label, minutiae_points))
            exists = cursor.fetchone()[0]
            if not exists:
                loop3 = loop3 + 1
                cursor.execute('''INSERT INTO MinutiaeData 
                                  (FingerprintID, GridLabel, MinutiaePoints, Distance, Angles) 
                                  VALUES (?, ?, ?, ?, ?)''', 
                               (fingerprint_id, grid_label, minutiae_points, json.dumps(triplet['Distances']), json.dumps(triplet['Angles'])))
    conn.commit()
    return loop1, loop2, loop3

def fetch_minutiae_data(conn):
    #meant to grab minutiae data from database and convert it to workable python data
    cursor = conn.cursor()
    query = "SELECT TripletID, GridLabel, MinutiaePoints, Distance, Angles FROM MinutiaeData"
    cursor.execute(query)
    return cursor.fetchall()


