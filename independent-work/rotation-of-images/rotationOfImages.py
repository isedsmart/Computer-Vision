import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

'''

This file can be ignored because the entire function and python code is written in jupyter notebooks.


'''


def rotateImg(img):
    # The image gets cut off if I implement it this way...
    imageWidth = len(img[0])
    imageHeight = len(img)
    allPixels = imageWidth*imageHeight
    row = []
    column = []
    rowToColumn = []
    columnToRow = []
    for h in range(imageHeight):
        column.append(img[h][0])

    for w in range(imageWidth):
        row.append(img[0][w])

    print(column)
    print(row)

    for index in reversed(range(imageWidth)):
        if (index < imageWidth):
            rowToColumn.insert(index, row[index])
        if (index < imageHeight):
            columnToRow.insert(index, column[index])

    # print(rowToColumn)
    # print(columnToRow)

    for h in range(imageWidth):
        for w in range(imageHeight):
            img[w][h] = columnToRow[w+h]

    print(img)

    pass








if __name__ == "__main__":
    test_image = cv2.imread("../../images/project3/testimage.pgm")
    monkey = cv2.imread("../../images/project3/demo.pgm")
    rotateImg(test_image)
