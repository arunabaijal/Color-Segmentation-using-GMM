from matplotlib import pyplot as plt
import numpy as np
import cv2
from copy import deepcopy
import os
import argparse

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.points = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.points.append([event.xdata, event.ydata])


def getroi(im, name, ch):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(name)
    line, = ax.plot([0], [0]) 
    linebuilder = LineBuilder(line)

    plt.imshow(im)
    plt.show()

    points = np.asarray(linebuilder.points, dtype = 'int32')

    x_left = int(min(linebuilder.xs[1:]))
    x_right = int(max(linebuilder.xs[1:]))
    y_up = int(min(linebuilder.ys[1:]))
    y_bottom = int(max(linebuilder.ys[1:]))

    final = np.zeros_like(im)
    m = np.zeros_like(im)
    cv2.fillConvexPoly(m, points, (255,255,255))
    mask = m[:, :, 0]

    final[(mask == 255)] = im[(mask == 255)]

    print(final.shape)
    cropped = final[y_up:y_bottom, x_left:x_right, :]

    plt.imshow(cropped)
    plt.show()
    # cv2.imshow("final", final)
    # # cv2.waitKey(10)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    if ch == 'r':        
        if not os.path.exists("Data/Red/Extracted/"):
            os.makedirs("Data/Red/Extracted/")
        cv2.imwrite("Data/Red/Extracted/" + name, cropped)


    elif ch == 'y':
        if not os.path.exists("Data/Yellow/Extracted/"):
            os.makedirs("Data/Yellow/Extracted/")
        cv2.imwrite("Data/Yellow/Extracted/" + name, cropped)

    elif ch == 'g':
        if not os.path.exists("Data/Green/Extracted/"):
            os.makedirs("Data/Green/Extracted/")
        cv2.imwrite("Data/Green/Extracted/" + name, cropped)

    else:
        print("Wrong choice")

def data_generation():
    ch = 'y'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    red_dir = os.path.join(current_dir, "Data/Red/Train")
    yellow_dir = os.path.join(current_dir, "Data/Yellow/Train") 
    green_dir = os.path.join(current_dir, "Data/Green/Train")
    
    count = 1
    if ch == 'r':
        for name in sorted(os.listdir(red_dir)):
            # if count < 19:
            #     count = count + 1
            #     continue
            im = cv2.imread(os.path.join(red_dir, name))
            count = count + 1

            getroi(im, name, ch)


    elif ch == 'y':
        for name in sorted(os.listdir(yellow_dir)):
            if count >= 24:
                count = count + 1
                break
            im = cv2.imread(os.path.join(yellow_dir, name))
            count = count + 1

            getroi(im, name, ch)

    elif ch == 'g':
        for name in sorted(os.listdir(green_dir)):
            # if count < 19:
            #     count = count + 1
            #     continue
            im = cv2.imread(os.path.join(green_dir, name))
            count = count + 1

            getroi(im, name, ch)

    else:
        print("Wrong choice")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Determine which data set to run for')
    # parser.add_argument('dataset', choices=['1', '2', '3'],
    #                     # help='Select which data set to run for, 1 or 2?')
    # args = parser.parse_args()
    data_generation()
