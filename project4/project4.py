import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


def Sobel_Edge_Horz(img):
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filtered_g_x = cv2.filter2D(img, cv2.CV_32F, g_x)
    return filtered_g_x

def Sobel_Edge_Vert(img):
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filtered_g_y = cv2.filter2D(img, cv2.CV_32F, g_y)
    return filtered_g_y

def Sobel_Edge(img):
    g = np.sqrt((Sobel_Edge_Horz(img)**2 + Sobel_Edge_Vert(img)**2))
    return g

def HarrisDetector(img,k = 0.04):
    '''
    Args:
    
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or color (your choice)
                (i recommmend greyscale)
    -   k: k value for Harris detector

    Returns:
    -   R: A numpy array of shape (m,n) containing R values of interest points
   '''

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grayImage = np.float32(grayImage)

    imageWidth = len(img[0])
    imageHeight = len(img)

    Ix = Sobel_Edge_Horz(grayImage)
    Iy = Sobel_Edge_Vert(grayImage)

    Ixx = np.multiply(Ix, Ix)
    Iyy = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)

    IxxConvolved = cv2.GaussianBlur(Ixx, (5, 5), 0)
    IyyConvolved = cv2.GaussianBlur(Iyy, (5, 5), 0)
    IxyConvolved = cv2.GaussianBlur(Ixy, (5, 5), 0)

    r = ((IxxConvolved * IyyConvolved) - (IxyConvolved ** 2)) - k * (IxxConvolved + IyyConvolved)**2


    # rList = []
    # for h in range(imageHeight):
    #     for w in range(imageWidth):
    #         IxxValue = IxxConvolved[h][w]
    #         IyyValue = IyyConvolved[h][w]
    #         IxyValue = IxyConvolved[h][w]
    #         a = [[IxxValue, IxyValue], [IxyValue, IyyValue]]
    #         a = np.float64(a)
    #         r = np.linalg.det(a) - k * (np.trace(a)**2)
    #         rList.append(r)


    rArray = np.asarray(r, dtype='float32')
    rArray = np.reshape(rArray, grayImage.shape)

    return rArray

def findMinValue(aList):

    numOfItems = len(aList)
    minValue = aList[0]
    for index in range(numOfItems):
        if index+1 < numOfItems and aList[index] < aList[index+1]:
            minValue = aList[index]

    return minValue



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
            if (Rvals[y][x]>0.01*Rmax):
                rList.append((x, y, Rvals[y][x]))

    sortedValues = sorted(rList, key=lambda tup: tup[2], reverse=True)

    lengthOfrList = len(rList)
    radii = []
    xIndex = 0
    yIndex = 1
    rIndex = 2

    for index in range(1, numPts*4):
        dList = []
        for counter in range(1, index):
            distance = np.sqrt(((sortedValues[index][xIndex]-sortedValues[index][xIndex])**2) + ((sortedValues[index-counter][yIndex]-sortedValues[index-counter][yIndex])**2))
            dList.append((sortedValues[index][xIndex], sortedValues[index][yIndex], distance))
        if len(dList) != 0:
            suppressedR = min(dList, key=lambda tup: tup[2])
            radii.append(suppressedR)

    sortedRadii = sorted(radii, key=lambda tup: tup[2], reverse=True)
    # print(sortedRadii)

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


if __name__ == "__main__":
    test_image = cv2.imread("testimage.pgm")
    # blur_test_image = cv2.GaussianBlur(test_image, (5, 5), 0)

    detector = HarrisDetector(test_image)

    SuppressNonMax(detector, 20)