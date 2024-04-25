#SOURCED AND EDITED
from image_enhancement import orientation
import math
import cv2 as cv
import numpy as np

def poincare_index_at(i, j, angles, tolerance):
    """
    compute the summation difference between the adjacent orientations such that the orientations is less then 90 degrees
    https://books.google.pl/books?id=1Wpx25D8qOwC&lpg=PA120&ots=9wRY0Rosb7&dq=poincare%20index%20fingerprint&hl=pl&pg=PA120#v=onepage&q=poincare%20index%20fingerprint&f=false
    :param i:
    :param j:
    :param angles:
    :param tolerance:
    :return:
    """
    cells = [(-1, -1), (-1, 0), (-1, 1),         # p1 p2 p3
            (0, 1),  (1, 1),  (1, 0),            # p8    p4
            (1, -1), (0, -1), (-1, -1)]          # p7 p6 p5

    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells]
    index = 0
    for k in range(0, 8):

        # calculate the difference
        difference = angles_around_index[k] - angles_around_index[k + 1]
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180

        index += difference

    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop"
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta"
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl"
    return "none"


def calculate_singularities(im, angles, tolerance, width, mask):
    singularities = []
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)

    # DELTA: RED, LOOP:ORAGNE, whorl:INK
    colors = {"loop" : (0, 0, 255), "delta" : (0, 128, 255), "whorl": (255, 153, 255)}
    for i in range(3, len(angles) - 2):  # Y
        for j in range(3, len(angles[i]) - 2):  # X
            # Mask any singularity outside of the mask
            mask_slice = mask[(i-2)*width:(i+3)*width, (j-2)*width:(j+3)*width]
            mask_flag = np.sum(mask_slice)
            if mask_flag == (width*5)**2:
                singularity_type = poincare_index_at(i, j, angles, tolerance)
                if singularity_type != "none":
                    # Store singularity type and center coordinates of the cell
                    singularities.append(((j+0.5)*width, (i+0.5)*width)) #just deleted ", singularity_type" as argument. Add back if needed
                    cv.rectangle(result, ((j+0)*width, (i+0)*width), ((j+1)*width, (i+1)*width), colors[singularity_type], 3)
    return result, singularities

def find_singularity_center(singularities):
    # Assuming singularities are provided in the order: bottom left, bottom right, top left, top right
    if singularities:
        x_coords = [singularities[0][0], singularities[1][0], singularities[2][0], singularities[3][0]]
        y_coords = [singularities[0][1], singularities[1][1], singularities[2][1], singularities[3][1]]
        
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        
        return (center_x, center_y)
    else:
        return None
    
def center_and_scale_fingerprint(img, center, target_size=(328, 356)):
    img_cx, img_cy = img.shape[1] // 2, img.shape[0] // 2
    singularity_x, singularity_y = center
    
    # Calculate translation to move singularity to image center
    translation_x, translation_y = img_cx - singularity_x, img_cy - singularity_y
    
    # Translate the image to center on the singularity
    M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    translated_img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    # Optional: scale the image if desired size is different from current size
    # This step would be necessary if you're looking to not just translate but also resize the image
    # scaled_img = cv.resize(translated_img, target_size)
    scaled_img = cv.resize(translated_img, target_size, interpolation=cv.INTER_AREA)
    return scaled_img  # or scaled_img if resizing

def merge_close_singularities(singularities, merge_distance=10):
    merged = []
    for s in singularities:
        x, y, s_type = s
        # Check if singularity is close to any already merged
        found_close = False
        for m in merged:
            mx, my, m_type = m
            if s_type == m_type and ((x - mx) ** 2 + (y - my) ** 2) ** 0.5 < merge_distance:
                found_close = True
                break
        if not found_close:
            merged.append(s)
    return merged


