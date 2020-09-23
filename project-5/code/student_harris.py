import cv2
import numpy as np
import matplotlib.pyplot as plt


def Sobel_Edge_Horz(img):
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filtered_g_x = cv2.filter2D(img, cv2.CV_32F, g_x)
    return filtered_g_x


def Sobel_Edge_Vert(img):
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filtered_g_y = cv2.filter2D(img, cv2.CV_32F, g_y)
    return filtered_g_y


def Sobel_Edge(img):
    g = np.sqrt((Sobel_Edge_Horz(img) ** 2 + Sobel_Edge_Vert(img) ** 2))
    return g


def HarrisDetector(img, k=0.04):
    '''
    Args:

    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or color (your choice)
                (i recommmend greyscale)
    -   k: k value for Harris detector

    Returns:
    -   R: A numpy array of shape (m,n) containing R values of interest points
   '''


    Ix = Sobel_Edge_Horz(img)
    Iy = Sobel_Edge_Vert(img)

    Ixx = np.multiply(Ix, Ix)
    Iyy = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)

    IxxConvolved = cv2.GaussianBlur(Ixx, (5, 5), 0)
    IyyConvolved = cv2.GaussianBlur(Iyy, (5, 5), 0)
    IxyConvolved = cv2.GaussianBlur(Ixy, (5, 5), 0)

    r = (IxxConvolved * IxyConvolved ** 2) - k * (IxxConvolved + IyyConvolved)**2

    rArray = np.asarray(r, dtype='float32')
    rArray = np.reshape(rArray, img.shape)

    return rArray


def SuppressNonMax(Rvals, numPts):
    '''
    Args:

    -   Rvals: A numpy array of shape (m,n,1), containing Harris response values
    -   numPts: the number of responses to return

    Returns:

     x: A numpy array of shape (N,) containing x-coordinates of interest points
     y: A numpy array of shape (N,) containing y-coordinates of interest points
     confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
   '''

    imageWidth = len(Rvals[0])
    imageHeight = len(Rvals)

    Rmax = Rvals.max()

    rList = []

    for y in range(imageHeight):
        for x in range(imageWidth):
            if np.any(Rvals[y][x] > 0.01 * Rmax):
                rList.append((x, y, Rvals[y][x]))

    sortedValues = sorted(rList, key=lambda tup: tup[2], reverse=True)

    radii = []
    xIndex = 0
    yIndex = 1
    rIndex = 2

    for index in range(1, numPts):
        dList = []
        for counter in range(1, index):
            if (index <= len(rList)):
                distance = np.sqrt(((sortedValues[index][xIndex] - sortedValues[index][xIndex]) ** 2) + (
                (sortedValues[index - counter][yIndex] - sortedValues[index - counter][yIndex]) ** 2))
                dList.append((sortedValues[index-counter][xIndex], sortedValues[index-counter][yIndex], distance))
            if len(dList) != 0:
                suppressedR = min(dList, key=lambda tup: tup[2])
                radii.append(suppressedR)


    sortedRadii = sorted(radii, key=lambda tup: tup[2], reverse=True)

    radiiX = []
    radiiY = []
    radiiX.append(sortedRadii[0][xIndex])
    radiiY.append(sortedRadii[0][yIndex])
    for number in range(numPts):
        radiiX.append(sortedRadii[number][xIndex])
        radiiY.append(sortedRadii[number][yIndex])

    npX = np.asarray(radiiX, dtype='int')
    npY = np.asarray(radiiY, dtype='int')

    return npX, npY


def get_interest_points(image, feature_width):
    """

    JR adds: to ensure compatability with project 4A, you simply need to use
    this function as a wrapper for your 4A code.  Guidelines below left
    for historical reference purposes.

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    k = 0.4
    responses = HarrisDetector(image, k)

    x, y = SuppressNonMax(responses, feature_width)

    return x, y

if __name__ == "__main__":
    test_image = cv2.imread("../../images/project3/testimage.pgm")
    # blur_test_image = cv2.GaussianBlur(test_image, (5, 5), 0)

    # detector = HarrisDetector(test_image)

    get_interest_points(test_image, 16)

    # print(detector)

    # SuppressNonMax(detector, 20)
