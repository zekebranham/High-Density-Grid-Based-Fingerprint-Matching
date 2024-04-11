
# Algorithm for finding minutiae triplets

import math
import json
from typing import List, Tuple
import itertools


def process_cascade2(grid_size_x, grid_size_y, image_width):
    #Gathers the data from the cascade 1 that is stored into a json file
    with open('cascade1_output.json', 'r') as f:
        cascade1_data = json.load(f)
    #Gets the IDs of the grids, positions for terminations and positions for bifurcations
    high_density_grid_ids = cascade1_data['high_density_grid_ids']
    term_positions = [MinutiaPointHD(x, y) for x, y in cascade1_data['minutiae']['terminations']]
    bif_positions = [MinutiaPointHD(x, y) for x, y in cascade1_data['minutiae']['bifurcations']]
    
    all_minutiae = term_positions + bif_positions
    #Total number of positions
    
    # Process each high-density grid
    for grid_id in high_density_grid_ids:
        grid_minutiae = filter_minutiae_by_grid(all_minutiae, grid_id, grid_size_x, grid_size_y, image_width)
        triplets = form_and_analyze_triplets_for_grid(grid_minutiae)

        # Print details for a sample of 3 triplets
        sample_triplets = triplets[:3]  # Get the first 3 triplets
        for i, triplet in enumerate(sample_triplets):
            print(f"Sample Triplet {i+1} in Grid {grid_id}:")
            print("  Minutiae Points:")
            for j, point in enumerate(triplet.minutiae):
                print(f"    Point {j+1}: (x={point.x}, y={point.y})")
            print("  Distances:")
            for j, distance in enumerate(triplet.distances):
                print(f"    Distance {j+1}-{(j+2)%3+1}: {distance:.2f}")
            print("  Angles (degrees):")
            for j, angle in enumerate(triplet.angles):
                print(f"    Angle at Point {j+1}: {angle:.2f}°")
            print()


class MinutiaPointHD:             
#Changes data type from tuple to a type of class MinutiaPoint
    def __init__(self, x, y):
        self.x = x
        self.y = y
    #Math functions for calculating features    
    def distance_to(self, other: 'MinutiaPoint') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Triplet:
    def __init__(self, minutiae: List[MinutiaPointHD]):
        self.minutiae = self._shift_clockwise(minutiae)
        self.distances = self._calculate_distances()
        self.angles = self._calculate_angles()

    def _shift_clockwise(self, minutiae: List[MinutiaPointHD]) -> List[MinutiaPointHD]:
        cx = sum([point.x for point in minutiae]) / len(minutiae)
        cy = sum([point.y for point in minutiae]) / len(minutiae)

        angles = [math.atan2(point.y - cy, point.x - cx) for point in minutiae]
        minutiae_and_angles = list(zip(minutiae, angles))
        
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
            distance = self.minutiae[i].distance_to(self.minutiae[next_i])
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


def form_and_analyze_triplets_for_grid(minutiae: List[MinutiaPointHD]) -> List[Triplet]:
    triplets = [Triplet(list(triplet_minutiae)) for triplet_minutiae in itertools.combinations(minutiae, 3)]
    return triplets

def process_cascade2(highest_density_grid_id, grid_size_x, grid_size_y, image_width):
    # Load the data saved by Cascade 1
    with open('cascade1_output.json', 'r') as f:
        cascade1_data = json.load(f)

    # Extract minutiae positions and convert them to MinutiaPoint instances
    term_positions = [MinutiaPointHD(x, y) for x, y in cascade1_data['minutiae']['terminations']]
    bif_positions = [MinutiaPointHD(x, y) for x, y in cascade1_data['minutiae']['bifurcations']]
    
    # Combine termination and bifurcation points
    all_minutiae = term_positions + bif_positions

    # Filter the minutiae for the highest-density grid
    grid_minutiae = filter_minutiae_by_grid(all_minutiae, highest_density_grid_id, grid_size_x, grid_size_y, image_width)

    # Form triplets from the filtered minutiae
    triplets = form_and_analyze_triplets_for_grid(grid_minutiae)

    # Select a sample of 3 triplets to display
    sample_triplets = triplets[:3]

    # Print details for each triplet in the sample
    for i, triplet in enumerate(sample_triplets):
        print(f"Sample Triplet {i+1} in Highest-Density Grid {highest_density_grid_id}:")
        print("  Minutiae Points:")
        for j, point in enumerate(triplet.minutiae):
            print(f"    Point {j+1}: (x={point.x}, y={point.y})")
        print("  Distances:")
        for j, distance in enumerate(triplet.distances):
            print(f"    Distance {j+1}-{(j+2)%3+1}: {distance:.2f}")
        print("  Angles (degrees):")
        for j, angle in enumerate(triplet.angles):
            print(f"    Angle at Point {j+1}: {angle:.2f}°")
        print()

def filter_minutiae_by_grid(minutiae: List[MinutiaPointHD], grid_id: int, grid_size_x: int, grid_size_y: int, image_width: int) -> List[MinutiaPointHD]:
    filtered_minutiae = []
    
    # Calculate number of grids per row
    grids_per_row = math.ceil(image_width / grid_size_x)

    for point in minutiae:
        # Determine the grid coordinates (x, y) for the current minutia point
        grid_x = point.x // grid_size_x
        grid_y = point.y // grid_size_y
        
        # Calculate the ID of the grid to which the current minutia point belongs
        current_grid_id = grid_y * grids_per_row + grid_x
        
        # If the current grid ID matches the specified grid_id, add the point to the filtered list
        if current_grid_id == grid_id:
            filtered_minutiae.append(point)

    return filtered_minutiae
