import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


def Gaussian_Blur(img):
    cutoff_frequency = 7
    filter = cv2.getGaussianKernel(ksize=cutoff_frequency * 4 + 1,
                                   sigma=cutoff_frequency)
    filter = np.dot(filter, filter.T)
    blur = cv2.filter2D(img, -1, filter)

    return blur

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

def myHoughLines(image, rho, theta, threshold):
    # image = input array
    # rho = float --> distance from pixel (it's a range, a sampling number)
    # theta = float --> angle from the pixel (a range, a sampling number)
    # threshold = int
    # return a vector of lines (rho, theta)

    imageArray = np.dsplit(image, 1)
    imageMultiArray = np.array(imageArray)
    imageWidth = len(imageMultiArray[0][0])
    imageHeight = len(imageMultiArray[0])

    thetaValues = np.linspace(0, math.pi, 1/theta) # A range of theta values to be tested for

    lines = []

    for numberOfRows in range(imageHeight):
        for lengthOfRow in range(imageWidth):
            currentPixelHeight = numberOfRows
            currentPixelWidth = lengthOfRow
            for theta in thetaValues:
                currentRho = currentPixelWidth * math.cos(theta) + currentPixelHeight * math.sin(theta)
                lines.append((round(currentRho, rho), theta))

    commonLines = Finding_Identical_Lines(lines)

    # print(lines)
    # print(commonLines)

    #=================================================================#

    # This code was for np.unique if I ended up using it

    # print(commonLines[0])

    # thresholdLines = []
    #
    # for element in range(len(commonLines[0])):
    #     if (commonLines[1][element] >= threshold):
    #         thresholdLines.append(commonLines[0][element])

    # print(thresholdLines)

    #==================================================================#

    thresholdLines = []

    for line in commonLines:
        if commonLines.get(line) >= threshold:
            thresholdLines.append(line)

    return thresholdLines


def Finding_Identical_Lines(lines):
    thresholdLines = {}
    for line in lines:
        if line in thresholdLines:
            thresholdLines[line] += 1
        else:
            thresholdLines[line] = 1

    # thresholdLines = np.unique(lines, False, False, True)
    return thresholdLines

# Part B ==================================================================== #

def Finding_Gradient(img):
    G_x = Sobel_Edge_Horz(img)
    G_y = Sobel_Edge_Vert(img)
    mag = np.sqrt(G_x**2 + G_y**2)
    theta = np.arctan2(G_y, G_x)
    return np.dstack((mag, theta))


def Non_Maximum_Supp(img):
    gradient = Finding_Gradient(img)
    mag = np.dsplit(gradient, 2)[0]
    theta = np.dsplit(gradient, 2)[1]
    thetaValues = np.linspace(0, 3*(math.pi)/4, 4)
    theta = np.digitize(theta, thetaValues)
    for pixel_height in range(len(mag[0])-1):
        for pixel_width in range(len(mag)-1):
            if theta[pixel_width][pixel_height][0] == 0:
                if mag[pixel_width][pixel_height][0] <= mag[pixel_width+1][pixel_height][0] or mag[pixel_width][pixel_height][0] <= mag[pixel_width-1][pixel_height][0]:
                    mag[pixel_width][pixel_height] = 0
            elif theta[pixel_width][pixel_height][0] == np.pi/2:
                if mag[pixel_width][pixel_height][0] <= mag[pixel_width-1][pixel_height][0] or mag[pixel_width][pixel_height][0] <= mag[pixel_width][pixel_height+1][0]:
                    mag[pixel_width][pixel_height] = 0
            elif theta[pixel_width][pixel_height][0] == np.pi/4:
                if mag[pixel_width][pixel_height][0] <= mag[pixel_width+1][pixel_height+1][0] or mag[pixel_width][pixel_height][0] <= mag[pixel_width-1][pixel_height-1][0]:
                    mag[pixel_width][pixel_height] = 0
            elif theta[pixel_width][pixel_height][0] == 3*(np.pi)/4:
                if mag[pixel_width][pixel_height][0] <= mag[pixel_width-1][pixel_height+1][0] or mag[pixel_width][pixel_height][0] <= mag[pixel_width+1][pixel_height-1][0]:
                    mag[pixel_width][pixel_height] = 0

    return mag

def Setting_Thresholds(img, t_l, t_h):
    for pixel_height in range(len(img[0])):
        for pixel_width in range(len(img)):
            if img[pixel_width][pixel_height][0] > t_h:
                img[pixel_width][pixel_height] = 1
            elif img[pixel_width][pixel_height][0] > t_l:
                img[pixel_width][pixel_height] = 0.5
            else:
                img[pixel_width][pixel_height] = 0

    return img

def Edge_Tracking(img):
    for pixel_height in range(len(img[0])):
        for pixel_width in range(len(img)):
            if pixel_width - 1 > 0 and pixel_width + 1 < len(img[pixel_width]-1) and pixel_height - 1 > 0 and pixel_height + 1 < len(img):
                if img[pixel_width][pixel_height][0] == 0.5:
                    W = img[pixel_width-1][pixel_height][0]
                    NW = img[pixel_width-1][pixel_height+1][0]
                    N = img[pixel_width][pixel_height+1][0]
                    NE = img[pixel_width+1][pixel_height+1][0]
                    E = img[pixel_width+1][pixel_height][0]
                    SE = img[pixel_width+1][pixel_height-1][0]
                    S = img[pixel_width][pixel_height-1][0]
                    SW = img[pixel_width-1][pixel_height-1][0]
                    if W == 1 or NW == 1 or N == 1 or NE == 1 or E == 1 or SE == 1 or S == 1 or SW == 1:
                        img[pixel_width][pixel_height] = 1

    return img

def Canny(img, threshold1, threshold2):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    sobel = Sobel_Edge(blur)
    grad = Finding_Gradient(sobel)
    Non_Max = Non_Maximum_Supp(grad)
    N_M = Non_Max / np.amax(Non_Max)
    threshold = Setting_Thresholds(N_M, threshold1, threshold2)
    edge = Edge_Tracking(threshold)
    return edge


if __name__ == "__main__":
    monkey = cv2.imread("../images/project3/demo.pgm")
    horz = cv2.imread("../images/project3/horz.pgm")
    vert = cv2.imread("../images/project3/vert.pgm")
    dots = cv2.imread("../images/project3/four-horiz-dots.pgm")
    test_image = cv2.imread("../images/project3/testimage.pgm")

    blur_test_image = cv2.GaussianBlur(test_image, (5, 5), 0)

    gaussian_blur_monkey = Gaussian_Blur(monkey)
    gaussian_blur_horz = Gaussian_Blur(horz)
    gaussian_blur_dots = Gaussian_Blur(dots)
    gaussian_blur_test_image = Gaussian_Blur(test_image)

    sobel_edge_horz = Sobel_Edge(gaussian_blur_horz)
    sobel_edge_dots = Sobel_Edge(gaussian_blur_dots)
    sobel_edge_monkey = Sobel_Edge(gaussian_blur_monkey)
    sobel_edge_test_image = Sobel_Edge(gaussian_blur_test_image)

    sobel_edge_VERT = Sobel_Edge_Vert(gaussian_blur_test_image)

    grad = Finding_Gradient(gaussian_blur_test_image)

    # Non_Maximum_Supp(test_image)

    # thresh = Setting_Thresholds(blur_test_image, 0.2, 0.3)

    # Edge_Tracking(thresh)

    img = Canny(blur_test_image, 0.2, 0.3)
    # plt.imshow(img.squeeze())
    # plt.imshow((np.clip(img, 0, 1)*255).astype(np.uint8))
    # plt.imshow(img.shape)
    # plt.show()

    # print(np.dsplit(grad, 1)[0])


    # dot_dimensions = int(np.hypot(len(sobel_edge_dots[0]), len(sobel_edge_dots[0][0])))
    # test_image_dimensions = int(np.hypot(len(sobel_edge_test_image[0]), len(sobel_edge_test_image[0][0])))

    # Hough Transform Testing...

    # print(myHoughLines(sobel_edge_test_image, test_image_dimensions, np.pi/180, 350))




