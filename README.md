# High-Density Grid-Based Fingerprint Matching
 Overview Approach:
 This algorithm involves taking a fingerprint template and dividing the fingerprint into grids of size
 y the amount of grids being size x. After diving into X amount of square grid segments of
 size Y, each grid will be numbered for distinguishability from 0-X.
 The next process will involve processing the entire fingerprint for an initial base minutiae
 detection, only ends and bifurcations. From there, each square segment labeled 0-X will be given
 an attribute of the amount of basic minutiae present within each square, and the coordinate
 locations of each grid in perspective of the entire fingerprint will be stored
 After this, the grid's minutiae density above the global average will selected as “high-density.”
 This leaves, for simplicity, 2/X remaining square grid segments. Within each grid, an
 algorithm for a more advanced minutiae detection approach will be implemented. These methods
 may include minutiae triplets, local structures, or extra minutiae. From the chosen method a
 matching score will be generated and will be unique to the fingerprint.
 Identification Applications:
 For identification, I feel that this algorithm may be more suited. With the above approach, we
 can use a proposed cascade approach, taken from a Viola-Jones facial recognition paper study.
 With the cascade approach, we take a 1:M (one-to-many) process where we have one template
 and compare the template to the many templates of a large database.
 1. The first cascade level would be to gather global high-density grid segment location
 coordinates. We compare the high-density coordinates of the template with the
 coordinates of each of the database’s templates.
 2. The second cascade level would be gathering the minutiae within each grid and ridge
 frequency within these grids and comparing the template across the many.
 3. The third cascade level would be an extension of this
 4. The fourth cascade level would be comparing global matching scores, local matching
 scores, local structures, etc
 In using the cascade approach with the high-density grid segmentation, we would dramatically
 decrease the target pool with each cascade level and significantly improve overall performance.
 How this would work:
 1. The system would take in an enhanced fingerprint image, and then divide the fingerprint image
 into grids numbered from 0-X where X is the total number of grids. The number labels on each
 grid will be a critical part of high-density segment distinction later on. The number of grids will
 need to be consistent across the entire system, stored data, and the probed fingerprint (the
 fingerprint trying to be identified).
 2. From there the system would then extract all the basic minutiae data within the entire
 fingerprint, and track the number of minutiae within each 0-X segment. After collecting each
segment’s number of minutiae, average the number of minutiae present in each segment
 (For example: if the total number of minutiae is 100 minutiae points, there would be 100/X minutiae
 points per segment on average) and apply a threshold that will only allow the further processing
 of segments that hold above a certain amount of minutiae. The segments that remain should hold
 above the average amount of minutiae, hopefully reducing the number of segments by well over
 half the original segments (this process of average and reduction could be repeated if the overall
 amount of remaining segments is too many). The remaining segments will be the high-density
 segments.
 3. The next step would be for the system to apply further advanced minutiae data extraction on
 the remaining high-density segments, the method of choice should be minutiae triplets unless a
 better alternative is given. A triplet is a set of three minutiae points, which together form a
 unique pattern based on their relative positions and angles. The choice of triplets is significant
 because it allows the matching process to focus on the structural relationships between minutiae
 points, which are less likely to be affected by partial prints, distortions, or common fingerprint
 features across different individuals.
 4. This step starts the actual identification process, the previous steps are for the probe
 fingerprint pre-processing. The system will take the labeled high-density segments and compare
 the labels to the labels that are in the database of stored fingerprints. For example, let's say
 there are only 9 total segments of the fingerprint labeled as segment 0 through segment 8 from left
 to right, and top to bottom (there should be much more to be practical). If the probe high-density
 segments are segments [2,6,9], the system would filter out all candidate fingerprints from the said
 database whose high-density segments are NOT [2,6,9], leaving only the remaining fingerprints
 that have the same high-density segments as the probe fingerprint.
 5. The next step compares the extracted triplet data from each of the probe’s high-density
 segments, to those of the remaining candidate fingerprints in the database. So the probe’s
 segments would have triplet data1 from segment 2, triplet data2 from segment 6, and triplet
 data3 from segment 6. This triplet data from the probe will be compared against the remaining
 candidates, so the probe’s data1 from segment 2 will be compared against candidate #1’s data1
 from its segment 2, and the probe’s data2 from segment 6 will be compared against a candidate
 #2’s data2 from its segment 6, and so on. This process would repeat for each remaining candidate
 from the first filtering process and will get a matching score from each probe to candidate
 comparison, and the best matching score would result in the match for the probe
