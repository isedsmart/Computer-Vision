#project1.py
import numpy as np
import matplotlib.pyplot as plt
# import cv2

def loadppm(filename):
    '''Given a filename, return a numpy array containing the ppm image
    input: a filename to a valid ascii ppm file
    output: a properly formatted 3d numpy array containing a separate 2d array
            for each color
    notes: be sure you test for the correct P3 header and use the dimensions and depth
            data from the header
            your code should also discard comment lines that begin with #
    '''
    arrayOfValues = []
    beginningValue = 0
    file = open(filename, "r")
    for line in file:
        splitLine = line.split()
        if (splitLine[beginningValue] != "P3" and splitLine[beginningValue] != "#"):
            if (len(splitLine) == 2):
                width = int(splitLine[0])
                height = int(splitLine[1])
            elif (len(splitLine) == 1):
                continue
            else:
                arrayOfValues.append(splitLine)
    allValues = []
    redList = []
    greenList = []
    blueList = []
    for row in arrayOfValues:
        for value in row:
            allValues.append(value)
    i = 0
    while (i < len(allValues)):
        redList.append(allValues[i])
        i += 3
    i = 0
    while (i <len(allValues)):
        greenList.append(allValues[i+1])
        i += 3
    i = 0
    while (i < len(allValues)):
        blueList.append(allValues[i+2])
        i += 3
    newRedList = []
    newGreenList = []
    newBlueList = []
    colorCounter = 0
    for i in range(0, height):
        redRow = []
        greenRow = []
        blueRow = []
        for j in range(0, width):
            colorCounter += 1
            redRow.append(redList[colorCounter - 1])
            greenRow.append(greenList[colorCounter - 1])
            blueRow.append(blueList[colorCounter - 1])
        newRedList.append(redRow)
        newGreenList.append(greenRow)
        newBlueList.append(blueRow)

    redArray = np.array(newRedList, dtype='uint8')
    greenArray = np.array(newGreenList, dtype='uint8')
    blueArray = np.array(newBlueList, dtype='uint8')

    rgbArray = np.dstack((redArray, greenArray, blueArray))

    return rgbArray

def GetGreenPixels(img):
    '''given a numpy 3d array containing an image, return the green channel'''

    array = np.dsplit(img, 1)
    multiArrays = np.array(array, dtype='uint8')
    greenList = [0, 255, 0]
    greenArray = np.array(greenList, dtype='uint8')
    greenChannel = []

    for row in range(len(multiArrays[0])):
        for i in range(len(multiArrays[0][0])):
           if (np.array_equal(multiArrays[0][0][i], greenArray)):
                greenChannel.append(multiArrays[0][0][i])
    greenOnly = []
    for value in greenChannel:
        greenOnly = np.array(value, dtype='uint8')
    greenChannel = np.dstack(greenOnly)

    return greenChannel

def GetBluePixels(img):
    '''given a numpy 3d array containing an image, return the blue channel'''
    array = np.dsplit(img, 1)
    multiArrays = np.array(array, dtype='uint8')
    blueList = [0, 0, 255]
    blueArray = np.array(blueList, dtype='uint8')
    blueChannel = []

    for row in range(len(multiArrays[0])):
        for i in range(len(multiArrays[0][0])):
           if (np.array_equal(multiArrays[0][0][i], blueArray)):
                blueChannel.append(multiArrays[0][0][i])
    blueOnly = []
    for value in blueChannel:
        blueOnly = np.array(value, dtype='uint8')
    blueChannel = np.dstack(blueOnly)

    return blueChannel

def GetRedPixels(img):
    '''given a numpy 3d array containing an image, return the red channel'''
    array = np.dsplit(img, 1)
    multiArrays = np.array(array, dtype='uint8')
    redList = [255, 0, 0]
    redArray = np.array(redList, dtype='uint8')
    redChannel = []

    for row in range(len(multiArrays[0])):
        for i in range(len(multiArrays[0][0])):
            if (np.array_equal(multiArrays[0][0][i], redArray)):
                redChannel.append(multiArrays[0][0][i])
    redOnly = []
    for value in redChannel:
        redOnly = np.array(value, dtype='uint8')
    redChannel = np.dstack(redOnly)

    return redChannel

def ConvertToGreyScale(img):
    '''
    given a numpy 3d array containing an image, return a greyscale image of it
    :param img: numpy 3d array
    :return: 3d numpy image that's greyscaled
    '''

    array = np.dsplit(img, 1)
    multiArrays = np.array(array, dtype='uint8')
    grayList = []
    width = len(multiArrays[0][0])
    height = len(multiArrays[0])
    RGBinfo = len(multiArrays[0][0][0])
    # print(multiArrays[0][0][0][0]) # individual numbers
    for numberOfRows in range(height):
        for lengthOfRow in range(width):
            tempRGBSum = 0
            tempGrayList = []
            for individualValue in range(RGBinfo):
                tempRGBSum += multiArrays[0][numberOfRows][lengthOfRow][individualValue]
            rgbAve = int(tempRGBSum/3)
            tempGrayList.append(rgbAve)
            tempGrayList.append(rgbAve)
            tempGrayList.append(rgbAve)
            grayList.append(tempGrayList)

    colorCounter = 0
    newGrayList = []
    for i in range(0, height):
        grayRow = []
        for j in range(0, width):
            colorCounter += 1
            grayRow.append(grayList[colorCounter - 1])
        newGrayList.append(grayRow)

    grayArray = np.array(newGrayList, dtype='uint8')
    return grayArray

def ThresholdingImage(img):
    '''
    given a numpy 3d array containing an image, return a black and white image of it.
    :param img: 3d numpy array
    :return: 3d numpy image that's black and white
    '''
    grey = ConvertToGreyScale(img)
    height = len(grey)
    width = len(grey[0])
    RGBinfo = len(grey[0][0])
    blackAndWhite = []
    for rowNum in range(height):
        for lengthOfRow in range(width):
            tempBlackAndWhiteList = []
            for individualValue in range(RGBinfo):
                if (grey[rowNum][lengthOfRow][individualValue] < 128):
                    tempBlackAndWhiteList.append(0)
                else:
                    tempBlackAndWhiteList.append(255)
            blackAndWhite.append(tempBlackAndWhiteList)

    colorCounter = 0
    newBlackAndWhiteList = []
    for i in range(0, height):
        blackAndWhiteRow = []
        for j in range(0, width):
            colorCounter += 1
            blackAndWhiteRow.append(blackAndWhite[colorCounter - 1])
        newBlackAndWhiteList.append(blackAndWhiteRow)

    blackAndWhiteArray = np.array(newBlackAndWhiteList, dtype='uint8')
    return blackAndWhiteArray

def HistogramEqualization(img):
    '''
    given a numpy 3d array containing an image, return a balanced image of it.
    :param img: 3d numpy array
    :return: an equalized, balanced image
    '''

    rangeOfValues = 256

    greyScale = ConvertToGreyScale(img)
    rgbPixel  = 3 # Accounted for every pixel as 3 pixels in the ConvertToGreyScale function
    height = len(greyScale)
    width = len(greyScale[0])

    allValues = []
    # print(greyScale)

    listOfValues = {}
    # Initializing the dictionary of values for the Histogram
    for value in range(rangeOfValues):
        listOfValues.update({value: 0})

    for rowNum in range(height):
        for lengthOfRow in range(width):
            for individualValue in greyScale[rowNum][lengthOfRow]:
                allValues.append(individualValue)

    # Calculating the Histogram
    for elementNumber in range(rangeOfValues):
        counterForNumberSeen = 0
        for value in allValues:
            if (elementNumber == value):
                counterForNumberSeen += 1
        listOfValues.update({elementNumber: int(counterForNumberSeen / rgbPixel)})


    # Calculating the Cumulative Distribution C(I) of the Histogram Values
    cdfValues = listOfValues.copy()
    lastSeenNumber = cdfValues.get(0)

    for elementNumber in range(1, rangeOfValues):
        currentNumber = cdfValues.get(elementNumber) + lastSeenNumber
        if (cdfValues.get(elementNumber) != 0):
            lastSeenNumber = currentNumber
            cdfValues.update({elementNumber: currentNumber})
        else:
            lastSeenNumber = currentNumber
            cdfValues.update({elementNumber: lastSeenNumber})

    # print(cdfValues)

    # Rescaling the greyscale values accordingly...

    indexOfLowestValue = smallestValue(cdfValues)
    numOfPixels = height * width
    newValues = {}
    for i in range(rangeOfValues):
        value = cdfValues.get(i)
        lowestValue = cdfValues.get(indexOfLowestValue)
        h = round(((value - lowestValue) / (numOfPixels - lowestValue) * (rangeOfValues-1)))
        newValues.update({i: h})

    # print(newValues)

    # Updating the Values

    updatedValues = updateValues(allValues, newValues)
    # print(updatedValues)

    totalNumOfValues = len(updatedValues)
    newUpdatedValues = []
    counter = 0
    while (counter < totalNumOfValues):
        tempValues = []
        for lengthOfRow in range(width):
            tempValues.append(updatedValues[counter])
            counter += 1
        newUpdatedValues.append(tempValues)

    # print(newUpdatedValues)

    # Reconstructing the 2D array

    colorCounter = 0
    new2DList = []
    twoDArray = newUpdatedValues
    for i in range(height):
        twoDArrayRow = []
        for j in range(width):
            colorCounter += 1
            twoDArrayRow.append(twoDArray[colorCounter - 1])
        new2DList.append(twoDArrayRow)

    new2DArray = np.array(new2DList, dtype='uint8')

    # print(new2DArray)
    return new2DArray

# def ConvertToMultiDimensionalArray(aListOfValues, numberOfDimensions):
#     threeDList = []
#     totalNumOfValues = len(aListOfValues)
#     interval = 0
#     for index in range(totalNumOfValues):
#         tempList = []
#         for counter in range(numberOfDimensions):
#             tempList.append(aListOfValues[index])
#         threeDList.append(tempList)
#         interval += 1
#
#     pass

def updateValues(oldList, newValues):
    '''
    Creates a new list which replaces the old values with the new values that are given
    :param oldList: an old list of values
    :param newValues: a dictionary of the indices and their corresponding values
    :return: a new list of values
    '''
    totalNumOfValues = len(oldList)
    rangeOfValues = 256
    newListOfValues = []
    for i in range(totalNumOfValues):
        for j in range(rangeOfValues):
            if (oldList[i] == j):
                newListOfValues.append(newValues.get(j))

    return newListOfValues


def smallestValue(dictionary):
    '''
    Finds the smallest nonzero value within a given dictionary and returns the index of that nonzero value
    :param dictionary: a dictionary with keys and values
    :return: the index at which the nonzero value exists
    '''
    elementNumber = 0
    numberFound = False
    while elementNumber < len(dictionary) and not numberFound:
        if (dictionary.get(elementNumber) != 0):
            smallestNumber = elementNumber
            numberFound = True
        else:
            elementNumber += 1

    return smallestNumber


if __name__ == "__main__":
    # put any command-line testing code you want here.
    #note this code in this block will only run if you run the module from the command line
  # (i.e. type "python3 project1.py" at the command prompt)
  # or within your IDE
  # it will NOT run if you simply import your code into the python shell.

  rgb = loadppm("../images/simple.ascii.ppm")
  zebra = loadppm("../images/zebra.ascii.ppm")
  checkerboard = loadppm("../images/checkers.ascii.ppm")

  # ConvertToGreyScale(rgb)
  plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  plt.imshow(checkerboard)
  plt.show()

  # equalizationRGB = HistogramEqualization(rgb)
  # equalizationZ = HistogramEqualization(zebra)

  # blackAndWhiteRGB = ThresholdingImage(rgb)
  # blackAndwhiteZ = ThresholdingImage(zebra)


  # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
  # plt.imshow(equalizationZ)
  # plt.imshow(equalizationRGB)
  # plt.show()

  # plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  # plt.imshow(blackAndWhiteRGB)
  # plt.show()

  # grey = ConvertToGreyScale(zebra)
  # plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  # plt.imshow(grey)
  # plt.show()

  # uncomment the lines below to test
  #plt.imshow(rgb)
  #plt.show()

  #rgb = loadppm("../images/zebra.ascii.ppm")
  #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis

  #rgb = loadppm("../images/simple.ascii.ppm")
  #green = GetGreenPixels(rgb)

  #you know the routine
  #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  #plt.imshow(green,cmap='gray', vmin=0, vmax=255)
  #plt.show()

  # red = GetRedPixels(rgb)
  # plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  # plt.imshow(red,cmap='gray', vmin=0, vmax=255)
  # plt.show()


  #blue = GetBluePixels(rgb)
  #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  #plt.imshow(blue,cmap='gray', vmin=0, vmax=255)
  #plt.show()

  #code to test greyscale conversions of the colored boxes and the zebra

  #code to create black/white monochrome image

  #code to create/test normalized greyscale image