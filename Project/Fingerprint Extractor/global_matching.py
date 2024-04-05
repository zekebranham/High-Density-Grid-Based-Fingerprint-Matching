import math

class MinutiaPoint:
    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation

def euclidean_distance(m1, m2):
    """Calculate the Euclidean distance between two minutiae points."""
    return math.sqrt((m1.x - m2.x)**2 + (m1.y - m2.y)**2)

def angular_difference(m1, m2):
    """Calculate the absolute angular difference between two minutiae points."""
    angle_diff = abs(m1.orientation - m2.orientation)
    angle_diff = min(angle_diff, 360 - angle_diff)  # Account for angular wrap-around
    return angle_diff

def match_minutiae(minutiae1, minutiae2, distance_threshold, angle_threshold):
    """
    Match two sets of minutiae based on Euclidean distance and angular difference.

    Args:
        minutiae1 (list of MinutiaPoint): Minutiae set from the first fingerprint.
        minutiae2 (list of MinutiaPoint): Minutiae set from the second fingerprint.
        distance_threshold (float): The maximum allowed distance between two matching minutiae.
        angle_threshold (float): The maximum allowed angular difference between two matching minutiae.

    Returns:
        int: The number of matching minutiae pairs found.
    """
    matching_pairs = 0  # Initialize the count of matching pairs

    for m1 in minutiae1:
        for m2 in minutiae2:
            # Calculate the Euclidean distance and angular difference between the minutiae
            distance = euclidean_distance(m1, m2)
            angle_diff = angular_difference(m1, m2)

            # Check if both distance and angle are within the thresholds
            if distance <= distance_threshold and angle_diff <= angle_threshold:
                matching_pairs += 1  # Count this pair as matching

    return matching_pairs



"""
ADD TO MAIN FOR TESTING
inutiae_img, raw_term_positions, raw_bif_positions = calculate_minutiaes(centered_img, mask, kernel_size=3, edge_margin=20)

    # Convert raw minutiae positions to MinutiaPoint objects
    term_positions = [MinutiaPoint(x, y, orientation) for x, y, orientation in raw_term_positions]
    bif_positions = [MinutiaPoint(x, y, orientation) for x, y, orientation in raw_bif_positions]

    # Combine termination and bifurcation positions for global matching
    all_minutiae = term_positions + bif_positions

    # Define distance and angle thresholds for matching
    distance_threshold = 15  # Adjust based on your requirements
    angle_threshold = 30    # Adjust based on your requirements

    # Perform global matching and print the matching score
    matching_score = match_minutiae(all_minutiae, all_minutiae, distance_threshold, angle_threshold)  # Comparing the same set for illustration
    print(f"Global Matching Score: {matching_score}")
"""