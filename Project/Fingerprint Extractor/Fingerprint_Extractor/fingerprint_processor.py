"""
Won't Need this file, the process is done within Cascade1Fingerprint_Processor.py


import cv2 as cv
import numpy as np
from glob import glob
import feature_extractor
from image_enhancement.normalization import normalize
from image_enhancement.segmentation import create_segmented_and_variance_images
from image_enhancement.orientation import calculate_angles
from image_enhancement.frequency import ridge_freq
from image_enhancement.gabor_filter import gabor_filter
from image_enhancement.skeletonize import skeletonize
from image_enhancement.crossing_number import calculate_minutiaes


def load_image(image_path):
    return cv.imread(image_path, cv.IMREAD_GRAYSCALE)

def process_fingerprint(image_path, output_dir):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Perform image enhancement and processing steps
    normalized_img = normalize(img, 100, 100)
    cv.imwrite(f'{output_dir}/01_normalized.jpg', normalized_img)
    segmented_img, norm_img, mask = create_segmented_and_variance_images(normalized_img, 16, 0.2)
    cv.imwrite(f'{output_dir}/02_segmented.jpg', segmented_img)
    angles = calculate_angles(norm_img, 16)
    freq = ridge_freq(norm_img, mask, angles, 16, 5, 5, 15)
    gabor_img = gabor_filter(norm_img, angles, freq)
    cv.imwrite(f'{output_dir}/03_gabor_filtered.jpg', gabor_img)
    skeleton_img = skeletonize(gabor_img)
    cv.imwrite(f'{output_dir}/04_skeleton.jpg', skeleton_img)
    minutiae_img = calculate_minutiaes(skeleton_img)

    # Return or save your final image
    cv.imwrite('processed_fingerprint.jpg', minutiae_img)
    # Return any additional data you need
    return minutiae_img

def divide_into_grids(image, grid_size_x, grid_size_y):
    grids = []
    height, width = image.shape
    for x in range(0, width, grid_size_x):
        for y in range(0, height, grid_size_y):
            grids.append(image[y:y+grid_size_y, x:x+grid_size_x])
    return grids

#def detect_basic_minutiae(grid):
    # Placeholder for basic minutiae detection logic
    #pass

def minutiae_counts(img, spuriousMinutiaeThresh=10, invertImage=False):
    extractor = feature_extractor.SimplifiedFingerprintFeatureExtractor()
    #extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if invertImage:
        img = 255 - img

    #extractor.skeletonize(img)
    #extractor.getTerminationBifurcation(img)
    #extractor.extract_minutiae_counts(img)

    # Directly count minutiae without further processing
    #minutiaeTermCount = np.sum(extractor.minutiaeTerm)
    #minutiaeBifCount = np.sum(extractor.minutiaeBif)
    #skel = extractor.skeletonize(img)
    minutiaeTermCount, minutiaeBifCount = extractor.getTerminationBifurcation(img)


    return minutiaeTermCount, minutiaeBifCount

if __name__ == "__main__":
    # Path to the fingerprint image file
    output_dir = r"C:\Users\14807\Desktop\VSCode\CSCI 158\Project"
    image_path = r"C:\Users\14807\Desktop\CSCI 158 Project\022\L\022_L1_0.bmp"
    
    # Load the image
    img = process_fingerprint(image_path, output_dir)
    
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not found.")
    else:
        # Extract and count minutiae
        #print("This Loading Worked")
        minutiaeTermCount, minutiaeBifCount = minutiae_counts(img, spuriousMinutiaeThresh=10, invertImage=False)
        
        # Print the counts
        print(f"Termination Points: {minutiaeTermCount}")
        print(f"Bifurcation Points: {minutiaeBifCount}") 
        """