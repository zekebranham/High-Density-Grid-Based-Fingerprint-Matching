import math
import numpy as np
import cv2 as cv
from image_enhancement.crossing_number import minutiae_at

def calculate_orientation(image, x, y, window_size):
    """Calculate the orientation of a minutia at (x, y) in the given image using a gradient-based method."""

    # Ensure window_size is odd and greater than 0
    if window_size <= 0:
        window_size = 3  # Minimum viable window size
    elif window_size % 2 == 0:
        window_size += 1  # Make window_size odd if it's even

    # Compute gradients using Sobel operators
    gx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    gy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)

    # Square gradients and their product
    gxx = gx**2
    gyy = gy**2
    gxy = gx * gy

    # Apply Gaussian smoothing to the square gradients and their product
    sxx = cv.GaussianBlur(gxx, (window_size, window_size), 0)
    syy = cv.GaussianBlur(gyy, (window_size, window_size), 0)
    sxy = cv.GaussianBlur(gxy, (window_size, window_size), 0)

    # Compute the orientation for each pixel
    orientation_matrix = 0.5 * np.arctan2(2 * sxy, (sxx - syy))

    # Ensure x and y are within the bounds of the orientation_matrix
    y = max(0, min(y, orientation_matrix.shape[0] - 1))
    x = max(0, min(x, orientation_matrix.shape[1] - 1))

    # Access the orientation value at the (y, x) location
    orientation = orientation_matrix[y, x]

    return orientation

# Update your MinutiaPoint class to calculate orientation during initialization
class MinutiaPointGM:
    def __init__(self, x, y, image, window_size=16):
        self.x = x
        self.y = y
        self.orientation = calculate_orientation(image, x, y, window_size)

def euclidean_distance(m1, m2):
    """Calculate the Euclidean distance between two minutiae points."""
    return math.sqrt((m1.x - m2.x)**2 + (m1.y - m2.y)**2)

def angular_difference(m1, m2):
    """Calculate the absolute angular difference between two minutiae points."""
    angle_diff = abs(m1.orientation - m2.orientation)
    angle_diff = min(angle_diff, 360 - angle_diff)  # Account for angular wrap-around
    return angle_diff

def match_minutiae(minutiae1, minutiae2, distance_threshold, angle_threshold):
    matching_pairs = 0
    total_comparisons = min(len(minutiae1), len(minutiae2))  # To normalize the score

    for m1 in minutiae1:
        for m2 in minutiae2:
            distance = euclidean_distance(m1, m2)
            angle_diff = angular_difference(m1, m2)
            if distance <= distance_threshold and angle_diff <= angle_threshold:
                matching_pairs += 1

    # Normalize the matching score to be between 0 and 1
    matching_score = matching_pairs / total_comparisons if total_comparisons > 0 else 0
    return matching_score


def calculate_minutiaesGM(im, mask, kernel_size=3, edge_margin=10, window_size=5):
    binary_image = np.zeros_like(im)
    binary_image[im < 10] = 1
    binary_image = binary_image.astype(np.int8)

    (height, width) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

    term_positions = []
    bif_positions = []

    for i in range(edge_margin, width - kernel_size // 2 - edge_margin):
        for j in range(edge_margin, height - kernel_size // 2 - edge_margin):
            if mask[j, i] == 0:
                continue

            minutiae_type = minutiae_at(binary_image, j, i, kernel_size)
            if minutiae_type in ["ending", "bifurcation"]:
                orientation = calculate_orientation(im, i, j, window_size)
                minutia = MinutiaPointGM(i, j, orientation)
                if minutiae_type == "ending":
                    term_positions.append(minutia)
                else:
                    bif_positions.append(minutia)

    return result, term_positions, bif_positions
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