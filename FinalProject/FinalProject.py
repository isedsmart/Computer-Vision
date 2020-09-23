import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def convertToGrayScale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def gaussianBlur(img):
    cutoff_frequency = 7
    gaussian = cv2.getGaussianKernel(ksize=cutoff_frequency * 4 + 1, sigma=cutoff_frequency)
    filter = np.dot(gaussian, gaussian.T)
    blur = cv2.filter2D(img, cv2.CV_32F, filter)
    return blur

def sobelFilterX(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    return sobelx

def sobelFilterY(img):
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return sobely

def findContour(img):
#     imgcopy = img.copy()
    contour, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # (_, contour, _) = cv2.findContours(image=threshold(img),
    #                  mode=cv2.RETR_EXTERNAL,
    #                  method=cv2.CHAIN_APPROX_SIMPLE)
    return contour

def threshold(img):
    img1_8bit = np.uint8(img * 255)
    # adaptive_threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    val, threshold = cv2.threshold(img1_8bit, 127, 255, cv2.THRESH_BINARY)
    return threshold

def blobDetector(img):
    blobDetect = cv2.SimpleBlobDetector()
    keypoints = blobDetect.detect(img)
    imageKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return imageKeyPoints

def blobDetectorParams(img):
    params = blobDetector(img)
    params.minThreshold = 10
    params.maxThreshold = 200

    params.filterByArea = True
    params.minArea = 2500

    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = True
    params.minConvexity = 0.07

    params.filterByInteria = True
    params.minInertiaRatio = 0.01

    ver = (cv2.__version__).split(".")
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    blob = blobDetector(detector)
    return blob

def morph_close(img):
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

def morph_open(img):
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def morph_erosion(img, iterNum):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=iterNum)
    return erosion

def morph_dilate(img, iterNum):
    kernel = np.ones((2, 2), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=iterNum)
    return dilate

def combo(img):
    grayim = convertToGrayScale(img)
    thresholdim = threshold(grayim)
    
    erosionkern = np.ones((5, 5), np.uint8)
    open1kern = np.ones((7, 7), np.uint8)
    open2kern = np.ones((9, 9), np.uint8)
    
    erodeim = cv2.erode(thresholdim, erosionkern, 1)
    openim = cv2.morphologyEx(erodeim, cv2.MORPH_OPEN, open1kern)
    open2im = cv2.morphologyEx(openim, cv2.MORPH_OPEN, open2kern)
    return open2im
    


if __name__ == "__main__":
    image1 = plt.imread('../FinalProject/images/planaria1.TIF')
    # grayImage1 = convertToGrayScale(image1)
    # threshold = threshold(grayImage1)
    # blur = gaussianBlur(threshold)
    # contours = findContour(blur)

    # plt.imshow(threshold, cmap="gray")
    # plt.show()

    # contour = cv2.findContours(image1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image1, contour, 0, (0,0,255), 6)

    # morphC = morph_close(threshold)
    # morphC = morph_close(morphC)
    # erode = morph_erosion(morphC, 3)
    # dilate = morph_dilate(erode, 12)
    # erode = morph_erosion(dilate, 16)
    # plt.imshow(erode, cmap='gray')
    # plt.show()

    # dilate = morph_dilate(morphC, 1)
    # plt.imshow(dilate, cmap='gray')
    # plt.show()

    # erode = morph_erosion(dilate, 3)
    # plt.imshow(erode, cmap='gray')
    # plt.show()

    # morphC = morph_close(morphC)
    # plt.imshow(morphC, cmap='gray')
    # plt.show()
    # morphC = morph_close(morphC)
    # plt.imshow(morphC, cmap='gray')
    # plt.show()

    # blobDetect = blobDetector(blur)

    # blobParams = blobDetectorParams(blur)
    # plt.imshow(blobParams)
    # plt.show()


    # image1x = sobelFilterX(grayImage1)
    # image1y = sobelFilterY(image1x)
    # plt.imshow(image1y, cmap="gray")
    # plt.show()

    # blobImage = blobDetector(threshold)
    # plt.imshow(blobImage, cmap="gray")
    # plt.show()

    # findContour(grayImage1)

