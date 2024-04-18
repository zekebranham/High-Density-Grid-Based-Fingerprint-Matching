import cv2 as cv
import numpy as np


def minutiae_at(pixels, i, j, kernel_size):
    """
    https://airccj.org/CSCP/vol7/csit76809.pdf pg93
    Crossing number methods is a really simple way to detect ridge endings and ridge bifurcations.
    Then the crossing number algorithm will look at 3x3 pixel blocks:

    if middle pixel is black (represents ridge):
    if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
    if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation

    :param pixels:
    :param i:
    :param j:
    :return:
    """
    # if middle pixel is black (represents ridge)
    if pixels[i][j] == 1:

        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1),        # p1 p2 p3
                   (0, 1),  (1, 1),  (1, 0),            # p8    p4
                  (1, -1), (0, -1), (-1, -1)]           # p7 p6 p5
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),                 # p1 p2   p3
                   (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0),               # p8      p4
                  (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]           # p7 p6   p5

        values = [pixels[i + l][j + k] for k, l in cells]

        # count crossing how many times it goes from 0 to 1
        crossings = 0
        for k in range(0, len(values)-1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2

        # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
        # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"

    return "none"


#edited to include an edge_margin (did not end up working, no change)
#def calculate_minutiaes(im, mask, kernel_size=3, edge_margin=10):
    # Adjust the binary image creation as before
    biniry_image = np.zeros_like(im)
    biniry_image[im < 10] = 1.0
    biniry_image = biniry_image.astype(np.int8)

    (y, x) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}

    term_positions = [] #Coordinates for term
    bif_positions = [] #Coordinates for bif

    # Adjust loop to respect edge_margin
    for i in range(edge_margin, x - kernel_size // 2 - edge_margin):
        for j in range(edge_margin, y - kernel_size // 2 - edge_margin):
            #Skip if outside the masked fingerprint region
            if mask[j, i] == 0:
                continue
            
            minutiae = minutiae_at(biniry_image, j, i, kernel_size)
            if minutiae == "ending":
                cv.circle(result, (i, j), radius=2, color=colors[minutiae], thickness=2)
                term_positions.append((i,j)) #Add position to list
            elif minutiae == "bifurcation":
                cv.circle(result, (i, j), radius=2, color=colors[minutiae], thickness=2)
                bif_positions.append((i, j))  # Add the position to the list

    return result, term_positions, bif_positions

#linescan minutiae detection rather than the old method
def calculate_minutiaes(im, mask, kernel_size=3, edge_margin=10):
    binary_image = np.zeros_like(im)
    binary_image[im < 10] = 1  # Assuming ridges are darker than the threshold of 10
    binary_image = binary_image.astype(np.int8)

    (height, width) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}

    term_positions, bif_positions = [], []

    # Iterate over rows
    for y in range(edge_margin, height - edge_margin):
        # Track first and last ridge pixel in the row
        first_ridge_x, last_ridge_x = None, None
        for x in range(edge_margin, width - edge_margin):
            if mask[y, x] == 0:  # Ignore if outside the fingerprint region
                continue
            if binary_image[y, x] == 1:  # Ridge pixel detected
                if first_ridge_x is None:
                    first_ridge_x = x
                last_ridge_x = x

        # Check if ridge pixels were detected in the row
        if first_ridge_x is not None and last_ridge_x is not None:
            # Now that you have first and last ridge pixels, process the row excluding these pixels
            # Process minutiae excluding the first and last detected ridge pixels
            for x in range((first_ridge_x) + 1, last_ridge_x):
                minutiae_type = minutiae_at(binary_image, y, x, kernel_size)
                if minutiae_type == "ending":
                    term_positions.append((x, y))
                elif minutiae_type == "bifurcation":
                    bif_positions.append((x, y))

    # Repeat similar logic for columns, iterating over columns and scanning rows within each column
    for x in range(edge_margin, width - edge_margin):
        first_ridge_y, last_ridge_y = None, None
        for y in range(edge_margin, height - edge_margin):
            if mask[y, x] == 0:
                continue
            if binary_image[y, x] == 1:
                if first_ridge_y is None:
                    first_ridge_y = y
                last_ridge_y = y

        # Check if ridge pixels were detected in the column
        if first_ridge_y is not None and last_ridge_y is not None:
            # Process minutiae excluding the first and last detected ridge pixels
            for y in range(first_ridge_y + 1, last_ridge_y):
                minutiae_type = minutiae_at(binary_image, y, x, kernel_size)
                if minutiae_type == "ending":
                    term_positions.append((x, y))
                elif minutiae_type == "bifurcation":
                    bif_positions.append((x, y))

    # Filter for unique positions since some minutiae might be detected in both row and column passes
    term_positions = list(set(term_positions))
    bif_positions = list(set(bif_positions))

    return result, term_positions, bif_positions

def remove_false_minutiae(skeleton_img, term_positions, bif_positions, mask=None):
        true_term_positions = []
        true_bif_positions = []

        # Helper function to check if a pixel is within the image bounds
        def in_bounds(x, y, shape):
            return 0 <= x < shape[1] and 0 <= y < shape[0]

        # Analyze termination points
        for x, y in term_positions:
            # Ensure the minutia is within the masked area if a mask is provided
            if mask is not None and mask[y, x] == 0:
                continue

            neighborhood = skeleton_img[max(0, y-1):y+2, max(0, x-1):x+2]
            # Count how many ridge pixels are in the neighborhood; expect 1 if true ending
            if np.sum(neighborhood == 0) <= 3:  # Adjust based on your image's representation
                true_term_positions.append((x, y))

        # Analyze bifurcation points
        for x, y in bif_positions:
            if mask is not None and mask[y, x] == 0:
                continue

            neighborhood = skeleton_img[max(0, y-1):y+2, max(0, x-1):x+2]
            # Count transitions; expect 3 for a true bifurcation
            transitions = 0
            indices = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
            for i in range(len(indices) - 1):
                y1, x1 = indices[i]
                y2, x2 = indices[i + 1]
                if in_bounds(x1, y1, neighborhood.shape) and in_bounds(x2, y2, neighborhood.shape):
                    if neighborhood[y1, x1] != neighborhood[y2, x2]:
                        transitions += 1
            if transitions == 6:  # This counts a transition twice (0 to 1 and 1 to 0), hence 3 transitions * 2
                true_bif_positions.append((x, y))

        return true_term_positions, true_bif_positions
