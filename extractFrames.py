from matplotlib import pyplot as plt
import numpy as np
import cv2
from copy import deepcopy
import os
import argparse
# from roi import getroi

def extractFrames():
    images = []

    cap = cv2.VideoCapture('detectbuoy.avi')
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    count = 1
    while (True):
        ret, im = cap.read()
        if ret == True:
            if count < 10:
                name = "00" + str(count)
            elif count < 100:
                name = "0" + str(count)
            else:
                name = str(count)

            if not os.path.exists("Frames"):
                os.makedirs("Frames")
            cv2.imwrite("Frames/frame" + name + ".png", im)
            count = count + 1
            images.append(im)
        else:
            cap.release()
            break

    return images

def organiseFrames():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.join(current_dir, "Frames")
    red_dir = os.path.join(current_dir, "Data/Red")
    yellow_dir = os.path.join(current_dir, "Data/Yellow") 
    green_dir = os.path.join(current_dir, "Data/Green")

    images = extractFrames()

    # yellow_names = []
    # # print(red_dir)
    # for name in sorted(os.listdir(yellow_dir)):
    #     yellow_names.append(name)

    # # yellow_count = 1
    # for i in range(len(yellow_names)):
    #     name = yellow_names[i]
    #     im = cv2.imread(os.path.join(yellow_dir, name))
    #     # print(name)
    #     # cv2.imshow("image", im)
    #     # cv2.waitKey(1)
    #     getroi(im, name)
    #     # ch = raw_input("Want to crop again? ")
    #     # if ch == 'y' or ch == 'Y':
    #     #     i = i - 1
    #     #     continue

    #     # yellow_count = yellow_count + 1

    #     # if yellow_count == 11:
    #     #     yellow_count = 1
    
    # cap = cv2.VideoCapture('detectbuoy.avi')
    # if (cap.isOpened() == False):
    #     print("Unable to read camera feed")

    # count = 1
    # while (True):
    #     ret, im = cap.read()
    #     if ret == True:
    #         # im = cv2.undistort(im, K, D)
    #         # img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    #         # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #         # im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    #         # im = cv2.GaussianBlur(im, (5,5), 5)
    #         if count < 10:
    #             name = "00" + str(count)
    #         elif count < 100:
    #             name = "0" + str(count)
    #         else:
    #             name = str(count)
    #         cv2.imwrite("Frames/frame" + name + ".png", im)
    #         count = count + 1
    #         images.append(im)
    #     else:
    #         cap.release()
    #         break

    red_count = 1
    for name in sorted(os.listdir(dir_name)):
        im = cv2.imread(os.path.join(dir_name, name))
        # cv2.imshow("image", im)
        # cv2.waitKey(1)
        if red_count > 7:
            if not os.path.exists("Data/Red/Test/"):
                os.makedirs("Data/Red/Test/")
            cv2.imwrite("Data/Red/Test/" + name, im)
        else: 
            if not os.path.exists("Data/Red/Train/"):
                os.makedirs("Data/Red/Train/")
            cv2.imwrite("Data/Red/Train/" + name, im)

        red_count = red_count + 1

        if red_count == 11:
            red_count = 1

    yellow_count = 1
    yellow_names = []
    # print(red_dir)
    for name in sorted(os.listdir(dir_name)):
        im = cv2.imread(os.path.join(dir_name, name))
        # print(name)
        # cv2.imshow("image", im)
        # cv2.waitKey(1)
        if yellow_count > 7:
            if not os.path.exists("Data/Yellow/Test/"):
                os.makedirs("Data/Yellow/Test/")
            cv2.imwrite("Data/Yellow/Test/" + name, im)
        else: 
            if not os.path.exists("Data/Yellow/Train/"):
                os.makedirs("Data/Yellow/Train/")
            cv2.imwrite("Data/Yellow/Train/" + name, im)

        yellow_count = yellow_count + 1

        if yellow_count == 11:
            yellow_count = 1

    green_count = 1
    # print(green_dir)
    for name in sorted(os.listdir(dir_name)):
        im = cv2.imread(os.path.join(dir_name, name))
        # cv2.imshow("image", im)
        # cv2.waitKey(1)
        if green_count > 7:
            if not os.path.exists("Data/Green/Test/"):
                os.makedirs("Data/Green/Test/")
            cv2.imwrite("Data/Green/Test/" + name, im)
        else: 
            if not os.path.exists("Data/Green/Train/"):
                os.makedirs("Data/Green/Train/")
            cv2.imwrite("Data/Green/Train/" + name, im)

        green_count = green_count + 1

        if green_count == 11:
            green_count = 1

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Determine which data set to run for')
    # parser.add_argument('dataset', choices=['1', '2'],
    #                     help='Select which data set to run for, 1 or 2?')
    # args = parser.parse_args()
    extractFrames()
    organiseFrames()