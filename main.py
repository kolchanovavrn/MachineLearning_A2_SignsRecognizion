import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
from PIL import Image

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0, 2):
        prefix = rootpath + '/' + "Images/" + format(c, '05d') + '/' # subdirectory for class
        print(prefix + 'GT-' + format(c, '05d') + '.csv')
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()

    #padding
    print("PADDING start")
    print("RESIZE start")
    for i in range(len(images)):
        print(i)
        new_shape = max(images[i].shape)
        res = cv2.resize(images[i], dsize=(new_shape, new_shape), interpolation=cv2.INTER_CUBIC)


        res = cv2.resize(res, dsize=(30, 30), interpolation=cv2.INTER_CUBIC)

        plt.imshow(res)
        images[i] = res
    print("PADDING end")
    print("RESIZE end")
    print(len(images))

    print("Label", labels[])
    return images, labels

trainImages, trainLabels = readTrafficSigns('GTSRB/Final_Training')

#new = do_pad(trainImages)
print(len(trainLabels), len(trainImages))
# plt.imshow(trainImages[2])
# plt.imshow(new[1])
plt.show()


