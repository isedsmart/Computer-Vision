import numpy as np
import matplotlib.pyplot as plt
import utils as u
import cv2

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can finish grading
   before the heat death of the universe.
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

  imageArray = np.dsplit(image, 1)
  imageMultiArray = np.array(imageArray)
  newFilteredArray = imageMultiArray.copy()
  imageRGBList = []
  imageWidth = len(imageMultiArray[0][0])
  imageHeight = len(imageMultiArray[0])
  imageRGBValues = len(imageMultiArray[0][0][0])

  filterWidth = len(filter[0])
  filterHeight = len(filter)

  halfFilterWidth = filterWidth // 2
  halfFilterHeight = filterHeight // 2

  # filterArray = np.asarray(filter, 1)
  # filterMultiArray = np.array(filterArray)

  # print(imageMultiArray)
  # print(imageMultiArray[0][0][0])

  imageConv = np.zeros(image.shape)

  # halfFilterHeight, imageHeight - halfFilterHeight
  # halfFilterWidth, imageWidth - halfFilterWidth

  newFilteredValues = []
  for numberOfRows in range(imageHeight + filterHeight):
    tempNewFilteredValues = []
    for lengthOfRow in range(imageWidth + filterWidth):
      # sum = 0
      for numberOfFilterRows in range(filterHeight):
        for lengthOfFilterRow in range(filterWidth):
          # sum = sum + filter[numberOfFilterRows][lengthOfFilterRow] * image[numberOfRows - halfFilterHeight + numberOfFilterRows][lengthOfRow - halfFilterWidth + lengthOfFilterRow]
          pixelOfImage = imageMultiArray[0][numberOfFilterRows][lengthOfFilterRow]
          pixelOfFilter = filter[numberOfFilterRows][lengthOfFilterRow]
          resultingPixel = pixelOfImage * pixelOfFilter
          tempNewFilteredValues.append(resultingPixel)
      newFilteredValues.append(tempNewFilteredValues)
      # imageConv[numberOfRows][lengthOfRow] = sum

  # filtered_image = np.array(imageConv)
  filtered_image = np.array(newFilteredValues)
  print(newFilteredValues)



  # The width and height are the same in this case
  # print("--------------")
  # print(filter)
  # print("--------------")
  # print(filter[0]) # [6 5 8 1 4 5 1 ... ] --> 29 values in total in one list...
  # print(filter[0][1])

  # filtered_image = np.array(m, n, c)

  return filtered_image


def convolveVersion2(image, filter):
  '''
    Convolves (multiplies the corresponding indices) of the image with a filter and returns the center value.
    It's assumed that the image has a big enough area so that the filter can take affect.
    :param image: the 3d array image
    :param filter: the 2d array filter
    :return: the new center value
    '''
  imageArray = np.dsplit(image, 1)
  imageMultiArray = np.array(imageArray)
  imageWidth = len(imageMultiArray[0][0])
  imageHeight = len(imageMultiArray[0])
  individualPixel = len(imageMultiArray[0][0][0])


  filterWidth = len(filter)
  filterHeight = len(filter[0])

  indexOfMiddleWidth = round(filterWidth / 2)
  indexOfMiddleHeight = 1

  convolvedList = []

  for numberOfRows in range(filterHeight):
    tempList = []
    for lengthOfRow in range(filterWidth):
      # for individualValue in range(individualPixel):
      pixelOfImage = imageMultiArray[0][numberOfRows][lengthOfRow]
      pixelOfFilter = filter[lengthOfRow]
      resultingPixel = resultingRGBValues(pixelOfImage, pixelOfFilter)
      tempList.append(resultingPixel)
  convolvedList.append(tempList)

  # print(convolvedList)

  # print(convolvedList[0][14][1])
  middlePixel = convolvedList[0][indexOfMiddleWidth][indexOfMiddleHeight]

  return middlePixel


def fakeConvolve(image, filter):
  '''
  Convolves the an area of the image with a filter and returns the center value.
  (The area of the image and the filter are the same size)
  :param image: the 3d array image
  :param filter: the 3d array filter
  :return: the new center value
  '''

  imageArray = np.dsplit(image, 1)
  imageMultiArray = np.array(imageArray)
  imageWidth = len(imageMultiArray[0][0])
  imageHeight = len(imageMultiArray[0])

  indexOfMiddleWidth = round(imageWidth / 2)
  indexOfMiddleHeight = round(imageHeight / 2)


  filterArray = np.dsplit(filter, 1)
  filterMultiArray = np.array(filterArray)

  convolvedList = []

  for numberOfRows in range(imageHeight):
    tempList = []
    for lengthOfRow in range(imageWidth):
      pixelOfImage = imageMultiArray[0][0][lengthOfRow]
      pixelOfImage2 = filterMultiArray[0][0][lengthOfRow]
      resultingPixel = resultingRGBValues(pixelOfImage, pixelOfImage2)
      tempList.append(resultingPixel)
    convolvedList.append(tempList)

  middlePixel = convolvedList[indexOfMiddleHeight][indexOfMiddleWidth]

  return middlePixel


def convertToList(rgbValue):
  '''
  Given a 1d numpy array, return a list of it
  :param rgbValue: a 1d numpy array
  :return: a list of those values
  '''

  listOfValues = []
  for value in rgbValue:
    listOfValues.append(value)

  return listOfValues


def resultingRGBValues(rgbValue, rgbValue2):
  '''
  Takes in two 1d arrays of rgbValues and multiplies their corresponding indices and returns a list of them
  :param rgbValue: 1d array of rgbValues
  :param rgbValue2: the second 1d array of rgbValues
  :return: a list of the resulting rgb values
  '''

  rgbValueLength = len(rgbValue)
  rgbValues = []

  for i in range(rgbValueLength):
    rgbValues.append(rgbValue[i] * rgbValue2[i])

  return rgbValues # [1, 2, 3] [3, 4, 3,] [5, 4, 3]


def getMiddleValueOfRGBValue(rgbValue, filterValues):
  '''
  Takes a 1d array (the rgb value of a pixel) and returns the middle number
  :param rgbValue: a 1d array of rgbValues
  :return: the middle value
  '''

  return rgbValue[1]


def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)utils
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###

  raise NotImplementedError('`create_hybrid_image` function in ' + 
    '`project2.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################
  pass
  # return low_frequencies, high_frequencies, hybrid_image

if __name__ == "__main__":
  image1 = u.load_image('../project2/images/dog.bmp')
  image2 = u.load_image('../project2/images/cat.bmp')

  plt.imshow(image2)

  identity_filter = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
  identity_image = my_imfilter(image2, identity_filter)
  plt.imshow(identity_image)
  # done = save_image('../results/identity_image.jpg', identity_image)

  # creating a filter
  # cutoff_frequency = 7
  # filter = cv2.getGaussianKernel(ksize=cutoff_frequency * 4 + 1,
  #                                sigma=cutoff_frequency)
  # filter = np.dot(filter, filter.T)

  # print(filter[0])
  # print(filter[1])
  # print("---")
  # print(filter[0][0])

  # let's take a look at the filter!
  # plt.figure(figsize=(4, 4))
  # plt.imshow(filter)

  # applying filter to image
  # blurry_dog = my_imfilter(image1, filter)
  # plt.figure()
  # plt.imshow((blurry_dog * 255).astype(np.uint8))
