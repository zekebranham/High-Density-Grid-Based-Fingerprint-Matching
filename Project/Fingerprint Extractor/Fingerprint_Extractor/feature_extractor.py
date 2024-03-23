import cv2 as cv
import numpy as np
import skimage.morphology

#zeke

class SimplifiedFingerprintFeatureExtractor:
    def __init__(self):
        self.minutiaeTerm = None
        self.minutiaeBif = None

    def getTerminationBifurcation(self, skel):
        # Check if 'skel' has an extra dimension and remove it
        if len(skel.shape) == 3 and skel.shape[2] == 1:
        # If 'skel' is 3D with a single channel, reduce it to 2D
            skel = skel[:, :, 0]
        elif len(skel.shape) == 3:
        # If 'skel' is truly a 3-channel image, convert to grayscale
            skel = cv.cvtColor(skel, cv.COLOR_BGR2GRAY)

        #print(f"skel shape before loop: {skel.shape}")
        # Assuming 'skel' is the skeletonized image.
        rows, cols = skel.shape
        
        #initialize mainutiae matricies
        minutiaeTerm = np.zeros(skel.shape, dtype=np.uint8)
        minutiaeBif = np.zeros(skel.shape, dtype=np.uint8)

        minutiaeTermPositions = [] #DELETE list to store positions of terminations
        minutiaeBifPositions = [] #DELETE list to store positions of bifurications

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if skel[i, j] == 255:  # Pixel is part of a ridge.
                    block = skel[i-1:i+2, j-1:j+2]
                    block_val = np.sum(block == 255)
                    if block_val == 3:
                        minutiaeTerm[i, j] = 1
                        minutiaeTermPositions.append((j,i)) #DELETE store the (x, y) position
                    elif block_val == 4:
                        minutiaeBif[i, j] = 1
                        minutiaeBifPositions.append((j,i)) #DELETE store the (x, y) position
        self.minutiaeTerm = minutiaeTerm
        self.minutiaeBif = minutiaeBif

        #DELETE "minmutiaeTermPositions, minutiaeBifPositions"
        return np.sum(minutiaeTerm), np.sum(minutiaeBif), minutiaeTermPositions, minutiaeBifPositions

    def extract_minutiae_counts(self, img):
        #skel = self.skeletonize(img)
        return self.getTerminationBifurcation(img)
