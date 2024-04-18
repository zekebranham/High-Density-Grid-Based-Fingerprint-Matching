import json
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler

#Verification Process
"""
def compare_results(grids1, grids2):
    # Convert lists to sets for easier comparison
    set1 = set(grids1)
    set2 = set(grids2)
    
    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Calculate the Jaccard Index
    if not union:  # Prevent division by zero if both sets are empty
        return 1.0 if not intersection else 0  # Return 1.0 if both are empty, 0 otherwise
    jaccard_index = len(intersection) / len(union)
    
    return jaccard_index

def match_and_process(image_path1, image_path2, output_dir):
    # Extract data and perform initial image processing
    result1 = process_image_and_extract_data(image_path1, output_dir)
    result2 = process_image_and_extract_data(image_path2, output_dir)

    if not result1 or not result2:
        print("One or both images could not be processed.")
        return

    # Unpack results to obtain grid counts and high-density grid IDs
    _, grid_counts1, high_density_grids1 = result1
    _, grid_counts2, high_density_grids2 = result2

    # Perform initial grid similarity check
    grid_similarity_score = compare_results(high_density_grids1, high_density_grids2)
    print(f"Grid Similarity Score: {grid_similarity_score:.2f}")

    if grid_similarity_score >= 0.95:
        print("High grid similarity detected. Proceeding with detailed analysis in Extraction Process 2.")
"""        
#Identification Process
    
def json_to_float_list(data):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError as e:
            print(f"HEREEEEEE Error converting JSON to floats: {e}")
            return []
    if isinstance(data, list):
        return [float(item) for item in data]
    raise TypeError(" HEREEEEE Input data must be a JSON string or list of numbers")

def json_to_points(data):
    """ Converts JSON string of list of lists into a list of numpy arrays. """
    if isinstance(data, str):
        data = json.loads(data)  # Parse JSON to get the list of lists
    if all(isinstance(item, list) for item in data):  # Ensure it's a list of lists
        return [np.array(point) for point in data]
    else:
        raise ValueError("HEREEEEEE Data is not a list of lists")

def transform_data(minutiae_points_json, distances_json, angles_json):
    """Transforms JSON string data into structured Python dictionaries."""
    minutiae_points = json_to_points(minutiae_points_json)
    distances = json_to_float_list(distances_json)
    angles = json_to_float_list(angles_json)
    
    # Assuming that the data is correctly formatted as lists of triplets
    if len(minutiae_points) % 3 == 0:
        return [{
            'MinutiaePoints': minutiae_points[i:i+3],
            'Distances': distances[i:i+3],
            'Angles': angles[i:i+3]
        } for i in range(0, len(minutiae_points), 3)]
    return []

def transform_probe_data(probe_data, index2):
    #print(f"probe_triplets before transformation: {probe_triplets}")
    all_transformed_triplets = []
    for triplet in probe_data.get(index2, []):
        probe_minutiae_points = json_to_points(triplet['MinutiaePoints'])
        probe_distances = json_to_float_list(triplet['Distances'])
        probe_angles = json_to_float_list(triplet['Angles'])
        if len(probe_minutiae_points) % 3 == 0:
            transformed_triplets = [{
                'MinutiaePoints': probe_minutiae_points[i:i+3],
                'Distances': probe_distances[i:i+3],
                'Angles': probe_angles[i:i+3]
            } for i in range(0, len(probe_minutiae_points), 3)]
            all_transformed_triplets.append(transformed_triplets)
    return all_transformed_triplets

def transform_data_to_dict(group_of_triplets):
    if group_of_triplets and isinstance(group_of_triplets[0], dict):
        single_triplet = group_of_triplets[0]
        minutiae_points = single_triplet['MinutiaePoints']
        distances = single_triplet['Distances']
        angles = single_triplet['Angles']

        # Return a single dictionary without splitting into triplets
        return {
            'MinutiaePoints': minutiae_points,
            'Distances': distances,
            'Angles': angles
        }
    else:
        print("Invalid data format or empty list, returning empty dictionary.")
        return {}

def match_probe_fingerprint(probe_data, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    best_match = None
    best_score = float('inf')
    cursor.execute("SELECT DISTINCT FingerprintID FROM MinutiaeData")
    fingerprint_ids = cursor.fetchall()

    for (fingerprint_id,) in fingerprint_ids:
        fingerprint_scores = []
        #Query SQL Database for Candidate Fingerprint
        cursor.execute("""
            SELECT GridLabel, MinutiaePoints, Distance, Angles 
            FROM MinutiaeData 
            WHERE FingerprintID = ?
        """, (fingerprint_id,))
        candidate_data = cursor.fetchall()
        candidate_data_list = []
        triplets_by_grid_label = {}
        j=-1
        print(f"NEW CANDIDATE ID: {fingerprint_id}")
                
        for candidate_grid_label, minutiae_points_json, distances_json, angles_json in candidate_data:
            candidate_triplets = transform_data(minutiae_points_json, distances_json, angles_json) #candidate_triplets holds each INDIVIDUAL triplet of Candidate
            candidate_data_list.append(candidate_triplets) #add triplet to list of all triplets
            # Append the triplets to the corresponding list in the dictionary
            if candidate_grid_label not in triplets_by_grid_label:
                triplets_by_grid_label[candidate_grid_label] = []
            triplets_by_grid_label[candidate_grid_label].append(candidate_triplets)
        triplets_count_by_grid_label = {label: len(triplets) for label, triplets in triplets_by_grid_label.items()} #list to use if you want to iterate by triplet count in candidate data
        for grid_label in probe_data:
            if grid_label in triplets_by_grid_label:
                probe_triplets = transform_probe_data(probe_data, grid_label) #probe_triplets is a list of all the probe triplets in that grid
                candidate_triplets = triplets_by_grid_label[grid_label] #candidate_triplets is a list of all the candidate triplets in that grid
                num_triplets = min(len(probe_triplets), len(candidate_triplets))
                for i in range(num_triplets):
                    probe_triplet_list = probe_triplets[i] #the single probe triplet for this iteration within this grid
                    candidate_triplet_list = candidate_triplets[i] #the single candidate triplet for this iteration within this grid

                    probe_triplet = transform_data_to_dict(probe_triplet_list) #transforms the probe_triplet from a list to a dict to be iterated
                    candidate_triplet = transform_data_to_dict(candidate_triplet_list) #same here, converts list to dict

                    probe_minutiae_points = probe_triplet['MinutiaePoints']
                    probe_distances = probe_triplet['Distances']
                    probe_angles = probe_triplet['Angles']
                    candidate_minutiae_points = candidate_triplet['MinutiaePoints']
                    candidate_distances = candidate_triplet['Distances']
                    candidate_angles = candidate_triplet['Angles']

                    probe_distances_scaled, candidate_distances_scaled = Scaler(probe_distances, candidate_distances)
                    pd = points_distance(probe_minutiae_points, candidate_minutiae_points)
                    dd = distance_difference(probe_distances_scaled, candidate_distances_scaled)
                    ad = angles_difference(probe_angles, candidate_angles)

                    similarity_score = calculate_similarity(pd, dd, ad)
                    fingerprint_scores.append(similarity_score)

            # Aggregate and compare scores
            if fingerprint_scores:
                total_score = sum(fingerprint_scores)
                if total_score < best_score:
                    best_score = total_score
                    best_match = fingerprint_id
        print(f"ID: {fingerprint_id} Similarity Score: {total_score}")
    conn.close()
    return (best_match, best_score) if best_match is not None else ("No Match Found", None)


#this scaler function uses the machine learning library "scikit-learn" that provides methods to reshape the data
#this is important because it normalizes our here data to ensure fair comparison between measurements 
def Scaler(probe_distances, candidate_distances):
    probe_distances = json_to_float_list(probe_distances) #converts JSON to a workable floating-point list
    candidate_distances = json_to_float_list(candidate_distances)
    scaler = StandardScaler()                              #This reshapes the distances to 2D arrays with distances as one column
    # Reshape data for scaling (scikit-learn expects 2D arrays)
    probe_distances = np.array(probe_distances).reshape(-1, 1)
    candidate_distances = np.array(candidate_distances).reshape(-1, 1)
    # Fit the scaler on the combined data to ensure consistent scaling
    scaler.fit(np.vstack([probe_distances, candidate_distances]))  #fit is what "scaler" is trained on, both the probe and candidate distances. 
                                                                #This training involves calculating the mean and standard deviation of all the data it sees. 
    # Transform the distances using the fitted scaler
    probe_distances_scaled = scaler.transform(probe_distances).flatten()    #This process adjusts the distances by subtracting the mean and dividing by the standard deviation, which results in a dataset with a mean of zero and a standard deviation of one.
    candidate_distances_scaled = scaler.transform(candidate_distances).flatten()    #Since the output from the scaler is a 2D array, but for further processing (like calculating differences), a 1D array is more practical, the transformed data is flattened back into a 1D format.
    return probe_distances_scaled, candidate_distances_scaled



def points_distance(probe_minutiae_points, candidate_minutiae_points):
    """Calculate the average Euclidean distance between corresponding points in two lists."""
    if len(probe_minutiae_points) != len(candidate_minutiae_points):
        raise ValueError("Lists of probe and candidate points must have the same length")

    distances = []
    for probe_point, candidate_point in zip(probe_minutiae_points, candidate_minutiae_points):
        # Ensure that both probe_point and candidate_point are lists or tuples of floats
        probe_point = np.array(probe_point, dtype=float)
        candidate_point = np.array(candidate_point, dtype=float)
        
        # Calculate Euclidean distance between the two points
        distance = np.linalg.norm(probe_point - candidate_point)
        distances.append(distance)

    # Return the average of all calculated distances
    average_distance = np.mean(distances)
    return average_distance

def angles_difference(probe_angles, candidate_angles):
    # Normalize angles for cyclical nature
    probe_angles = np.radians(probe_angles)
    candidate_angles = np.radians(candidate_angles)
    angles_difference = np.abs(np.arctan2(np.sin(probe_angles - candidate_angles), np.cos(probe_angles - candidate_angles)))
    angles_difference = np.mean(angles_difference)
    return angles_difference

def distance_difference(probe_distances_scaled, candidate_distances_scaled):
    # Calculate mean differences for distances and angles
    distances_difference = np.mean(np.abs(probe_distances_scaled - candidate_distances_scaled))
    return distances_difference

def calculate_similarity(points_distance, distances_difference, angles_difference):
    # Initialize the StandardScaler
    # Compute similarity score as a weighted sum of differences
    weight_points = 1  # Adjust these weights as needed
    weight_distances = 1
    weight_angles = 1
    
    similarity_score = (weight_points * points_distance +
                        weight_distances * distances_difference +
                        weight_angles * angles_difference)
    return similarity_score

