import math
import json
from typing import List

class MinutiaPointHD:             
#Changes data type from tuple to a type of class MinutiaPoint
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __hash__(self):
        # Define the hash based on the immutable properties (x and y coordinates)
        return hash((self.x, self.y))

    def __eq__(self, other):
        # Ensure equality comparison is based on the same properties used in hashing
        return isinstance(other, self.__class__) and self.x == other.x and self.y == other.y
    #Math functions for calculating features    
    def distance_to(self, other: 'MinutiaPointHD') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2) # could add **0.5 if we want

def process_extraction2(grid_size_x, grid_size_y, image_width):
    #Gathers the data from the ExtractionONE that is stored into a json file
    with open('ExtractionONE_output.json', 'r') as f:
        ExtractionONE_data = json.load(f)
    #Gets the IDs of the grids, positions for terminations and positions for bifurcations
    high_density_grid_ids = ExtractionONE_data['high_density_grid_ids']
    term_positions = [MinutiaPointHD(x, y) for x, y in ExtractionONE_data['minutiae']['terminations']]
    bif_positions = [MinutiaPointHD(x, y) for x, y in ExtractionONE_data['minutiae']['bifurcations']]
    
    all_minutiae = term_positions + bif_positions
    triplets_data = {}
    #Total number of positions
    
    # Process each high-density grid
    
    for grid_id in high_density_grid_ids:
        loop1 = 0
        loop2 = 0
        grid_minutiae = filter_minutiae_by_grid(all_minutiae, grid_id, grid_size_x, grid_size_y, image_width)
        triplets = form_and_analyze_triplets_for_grid(grid_minutiae)

        # Store triplet data for the grid
        triplets_data[grid_id] = {
            'GridLabel': grid_id,  # This is the new part
            'Triplets': [{  # Wrap your triplet dicts in a 'Triplets' key
                'MinutiaePoints': [(single_minutia.x, single_minutia.y) for single_minutia in triplet.minutiaeList],
                'Distances': triplet.distances,
                'Angles': triplet.angles
            } for triplet in triplets]
        }
        # Print details for a sample of 3 triplets
        sample_triplets = triplets[:3]  # Get the first 3 triplets
    """
        for i, triplet in enumerate(sample_triplets):
            print(f"Sample Triplet {i+1} in Grid {grid_id}:")
            print("  Minutiae Points:")
            for j, point in enumerate(triplet.minutiaeList):
                print(f"    Point {j+1}: (x={point.x}, y={point.y})")
            print("  Distances:")
            for j, distance in enumerate(triplet.distances):
                print(f"    Distance {j+1}-{(j+2)%3+1}: {distance:.2f}")
            print("  Angles (degrees):")
            for j, angle in enumerate(triplet.angles):
                print(f"    Angle at Point {j+1}: {angle:.2f}Â°")
            print()
    """
    return triplets_data



class Triplet:
    def __init__(self, minutiaeList: List[MinutiaPointHD]):
        self.minutiaeList = minutiaeList
        self.minutiaeList = self._shift_clockwise(minutiaeList)
        self.distances = self._calculate_distances()
        self.angles = self._calculate_angles()

    def _shift_clockwise(self, minutiaeList: List[MinutiaPointHD]) -> List[MinutiaPointHD]:
        cx = sum([point.x for point in minutiaeList]) / len(minutiaeList)
        cy = sum([point.y for point in minutiaeList]) / len(minutiaeList)

        angles = [math.atan2(point.y - cy, point.x - cx) for point in minutiaeList]
        minutiae_and_angles = list(zip(minutiaeList, angles))
        
        # Sort minutiae by angle in ascending order
        sorted_minutiae_and_angles = sorted(minutiae_and_angles, key=lambda ma: ma[1])

        # Extract sorted minutiae
        sorted_minutiae = [ma[0] for ma in sorted_minutiae_and_angles]

        return sorted_minutiae
        
        #Math functions for calculating features
    def _calculate_distances(self) -> List[float]:
        distances = []
        for i in range(3):
            next_i = (i + 1) % 3
            distance = self.minutiaeList[i].distance_to(self.minutiaeList[next_i])
            distances.append(distance)
        return distances
    
    def _calculate_angles(self) -> List[float]:
        angles = []
        for i in range(3):
            a = self.distances[i]
            b = self.distances[(i + 1) % 3]
            c = self.distances[(i + 2) % 3]
            # Clamp the cosine value between -1 and 1 to avoid math domain errors
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_angle = max(-1, min(cos_angle, 1))  # Clamping
            angle = math.acos(cos_angle)
            angles.append(math.degrees(angle))
        return angles

# PREFERRED Attempts to use unique triplets only, tracking previously used minutiae
def form_and_analyze_triplets_for_grid(minutiaeList: List[MinutiaPointHD]) -> List[Triplet]:
    used_minutiae = set() #track used minutiae points
    triplets = []

    for single_minutia in minutiaeList:
        if single_minutia in used_minutiae:
            continue #skip if minutiae is already used

        #find the two nearest neighbors that haven't been used
        distances = sorted([(single_minutia.distance_to(m), m) for m in minutiaeList if m != single_minutia and m not in used_minutiae], key=lambda x: x[0])
        nearest_neighbors = distances[:2]  # Get two nearest neighbors
        
        if len(nearest_neighbors) == 2:  # Ensure there are enough neighbors
            triplet = Triplet([single_minutia] + [m for _, m in nearest_neighbors])
            triplets.append(triplet)
            #Mark the minutiae in the triplet as used
            used_minutiae.add(single_minutia) #add the current minutiae to the list
            used_minutiae.update([m for _ , m in nearest_neighbors]) #add neighbors

    return triplets

def process_high_density_grids():
    with open('ExtractionONE_output.json', 'r') as f:
        ExtractionONE_data = json.load(f)

    high_density_grid_ids = ExtractionONE_data['high_density_grid_ids']
    term_positions = [MinutiaPointHD(x, y) for x, y in ExtractionONE_data['minutiae']['terminations']]
    bif_positions = [MinutiaPointHD(x, y) for x, y in ExtractionONE_data['minutiae']['bifurcations']]
    
    all_minutiae = term_positions + bif_positions
    
    for grid_id in high_density_grid_ids:
    # filters the  minutiae to go into its corresponding grids
        grid_minutiae = filter_minutiae_by_grid(all_minutiae, grid_id)
        triplets = form_and_analyze_triplets_for_grid(grid_minutiae)
        # Creates the triplets
    
def filter_minutiae_by_grid(minutiaeList: List[MinutiaPointHD], grid_id: int, grid_size_x: int, grid_size_y: int, image_width: int) -> List[MinutiaPointHD]:
    filtered_minutiae = []
    
    # Calculate number of grids per row
    grids_per_row = math.ceil(image_width / grid_size_x)

    for point in minutiaeList:
        # Determine the grid coordinates (x, y) for the current minutia point
        grid_x = point.x // grid_size_x
        grid_y = point.y // grid_size_y
        
        # Calculate the ID of the grid to which the current minutia point belongs
        current_grid_id = grid_y * grids_per_row + grid_x
        
        # If the current grid ID matches the specified grid_id, add the point to the filtered list
        if current_grid_id == grid_id:
            filtered_minutiae.append(point)

    return filtered_minutiae