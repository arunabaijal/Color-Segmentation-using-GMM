#!/usr/bin/env python
# import imageio
# import matplotlib.animation as ani
# import matplotlib.cm as cmx
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal
import os
import numpy as np
import cv2
import sys
import glob
import math

# from matplotlib.patches import Ellipse
# from PIL import Image
# from sklearn import datasets
from sklearn.cluster import KMeans
from scipy import stats

def gaussian_fit(xdata,ydata):
    mu = np.sum(xdata*ydata)/np.sum(ydata)
    sigma = np.sqrt(np.abs(np.sum((xdata-mu)**2*ydata)/np.sum(ydata)))
    return mu, sigma

def get_histogram(data, ch):
    p, q, r = data.shape
      
    # calculate histogram of image
    histb = cv2.calcHist([data],[0],None,[256],[0,256])
    histg = cv2.calcHist([data],[1],None,[256],[0,256])
    histr = cv2.calcHist([data],[2],None,[256],[0,256])
    # ignore black values
    histr[0] = 0 
    histg[0] = 0
    histb[0] = 0

    # Generate gaussin fit for three channels
    x = np.arange(0, 256, 1) 
    mu_r, sigma_r = gaussian_fit(np.reshape(x, (256, 1)), np.reshape(histr, (256, 1)))
    y_r = norm.pdf(x, mu_r, sigma_r)
    mu_g, sigma_g = gaussian_fit(np.reshape(x, (256, 1)), np.reshape(histg, (256, 1)))
    y_g = norm.pdf(x, mu_g, sigma_g)
    mu_b, sigma_b = gaussian_fit(np.reshape(x, (256, 1)), np.reshape(histb, (256, 1)))
    y_b = norm.pdf(x, mu_b, sigma_b)


    fig, axs = plt.subplots(2)
    fig.suptitle('Display graphs')
    axs[0].plot(x, y_r, 'r', label="calculated")
    axs[0].plot(x, y_g, 'g', label="calculated")
    axs[0].plot(x, y_b, 'b', label="calculated")
    axs[1].plot(histr, 'r')
    axs[1].plot(histg, 'g')
    axs[1].plot(histb, 'b')

    plt.show() 

# Read train images for different categories and display histogram
def get_1Dgaussian(ch = 'r'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    red_dir = os.path.join(current_dir, "Data/Red/Extracted")
    yellow_dir = os.path.join(current_dir, "Data/Yellow/Extracted") 
    green_dir = os.path.join(current_dir, "Data/Green/Extracted")
    
    first = True
    count = 1
    if ch == 'r':
        for name in sorted(os.listdir(red_dir)):
            im = cv2.imread(os.path.join(red_dir, name))
            im = cv2.resize(im, (40, 40), interpolation = cv2.INTER_AREA)
            count = count + 1
            if first:
                data = im
                first = False
            else:
                data = np.column_stack((data, im))

        get_histogram(data, ch)


    elif ch == 'y':
        for name in sorted(os.listdir(yellow_dir)):
            im = cv2.imread(os.path.join(yellow_dir, name))
            im = cv2.resize(im, (40, 40), interpolation = cv2.INTER_AREA)
            count = count + 1
            if first:
                data = im
                first = False
            else:
                data = np.column_stack((data, im))

        get_histogram(data, ch)

    elif ch == 'g':
        for name in sorted(os.listdir(green_dir)):
            im = cv2.imread(os.path.join(green_dir, name))
            im = cv2.resize(im, (40, 40), interpolation = cv2.INTER_AREA)
            count = count + 1
            if first:
                data = im
                first = False
            else:
                data = np.column_stack((data, im))

        get_histogram(data, ch)

    else:
        print("Wrong choice")

def generate_gaussian(image_path):
    image = cv2.imread(image_path)
    red_pts = image[:,:,2]
    m_r, s_r = stats.norm.fit(red_pts)
    green_pts = image[:,:,1]
    m_g, s_g = stats.norm.fit(green_pts)
    blue_pts = image[:,:,0]
    m_b, s_b = stats.norm.fit(blue_pts)
    return [m_r, s_r, m_g, s_g, m_b, s_b]

def normal_dist_3D(mean):
    mu = np.mean(mean)
    cov_matrix = np.zeros((len(mean), len(mean)))
    cov_matrix[0][0] = np.sum((mean[:,0] - mu)**2) / len(mean)
    cov_matrix[1][1] = np.sum((mean[:,1] - mu)**2) / len(mean)
    cov_matrix[2][2] = np.sum((mean[:,2] - mu)**2) / len(mean)
    
    cov_matrix[0][1] = np.sum((mean[:,0] - mu)*(mean[:,1] - mu)) / len(mean)
    cov_matrix[0][2] = np.sum((mean[:,0] - mu)*(mean[:,2] - mu)) / len(mean)
    cov_matrix[1][0] = np.sum((mean[:,1] - mu)*(mean[:,0] - mu)) / len(mean)
    cov_matrix[1][2] = np.sum((mean[:,1] - mu)*(mean[:,2] - mu)) / len(mean)
    cov_matrix[2][0] = np.sum((mean[:,2] - mu)*(mean[:,0] - mu)) / len(mean)
    cov_matrix[2][1] = np.sum((mean[:,2] - mu)*(mean[:,1] - mu)) / len(mean)
    
    return mu, cov_matrix
    
def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    # print("covariance",cov)
    # print("diff", diff)
    # if np.linalg.det(cov) == 0:
    #     return np.zeros(X.shape[0]).reshape(-1, 1)
    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)

def initialize_clusters(X, n_clusters):
    clusters = []
    idx = np.arange(X.shape[0])
    
    # We use the KMeans centroids to initialise the GMM
    
    kmeans = KMeans().fit(X)
    mu_k = kmeans.cluster_centers_
    
    for i in range(n_clusters):
        clusters.append({
            'pi_k': 1.0 / n_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(X.shape[1], dtype=np.float64)*30
        })
    
    return clusters


def expectation_step(X, clusters):
    totals = np.zeros((X.shape[0], 1), dtype=np.float64)
    
    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        
        gamma_nk = (pi_k * gaussian(X, mu_k, cov_k)).astype(np.float64)
        # print(gamma_nk)
        for i in range(X.shape[0]):
            totals[i] += gamma_nk[i]
        
        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals
    
    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']


def maximization_step(X, clusters):
    N = float(X.shape[0])
    
    for cluster in clusters:
        gamma_nk = cluster['gamma_nk']
        cov_k = np.zeros((X.shape[1], X.shape[1]))
        
        N_k = np.sum(gamma_nk, axis=0)
        
        pi_k = N_k / N
        mu_k = np.sum(gamma_nk * X, axis=0) / N_k
        
        for j in range(X.shape[0]):
            diff = (X[j] - mu_k).reshape(-1, 1)
            cov_k += gamma_nk[j] * np.dot(diff, diff.T)
        
        cov_k /= N_k
        
        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k
        


def get_likelihood(X, clusters):
    likelihood = []
    sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
    return np.sum(sample_likelihoods), sample_likelihoods


def train_gmm(X, n_clusters, n_epochs, clusters, flag):
    if flag:
        clusters = clusters
    else:
        clusters = initialize_clusters(X, n_clusters)
    likelihoods = np.zeros((n_epochs,))
    scores = np.zeros((X.shape[0], n_clusters))
    history = []
    
    for i in range(n_epochs):
        clusters_snapshot = []
        
        # This is just for our later use in the graphs
        for cluster in clusters:
            clusters_snapshot.append({
                'mu_k': cluster['mu_k'].copy(),
                'cov_k': cluster['cov_k'].copy()
            })
        
        history.append(clusters_snapshot)
        
        expectation_step(X, clusters)
        maximization_step(X, clusters)
        
        likelihood, sample_likelihoods = get_likelihood(X, clusters)
        likelihoods[i] = likelihood
        
        print('Epoch: ', i + 1, 'Likelihood: ', likelihood)
    
    for i, cluster in enumerate(clusters):
        scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)
    
    return clusters, likelihoods, scores, sample_likelihoods, history

def create_pdf(params, X, img, shape, name):
    # seg_image = np.zeros(shape)

    seg_o = np.zeros((shape[0],shape[1]),dtype=np.uint8)
    seg_y = np.zeros((shape[0],shape[1]),dtype=np.uint8)
    seg_g = np.zeros((shape[0],shape[1]),dtype=np.uint8)
    print("seg:", seg_g.shape)

    pdf = np.zeros((len(X),3))
    x = np.array(X, dtype='uint8')

    # get pixel probabilities for all clusters
    for i, clusters in enumerate(params):
        for cluster in clusters:
            y_r = multivariate_normal.pdf(x, cluster['mu_k'], cluster['cov_k'])
            pdf[:,i] = pdf[:,i] + cluster['pi_k'] * y_r

    # Display generated GMM models
    # fig1 = plt.figure()
    # ax1 = fig1.gca()
    # ax1.set_title("Red")
    # ax1.plot(x[:,0], pdf[:,0], 'r')
    # ax1.plot(x[:,1], pdf[:,0], 'g')
    # ax1.plot(x[:,2], pdf[:,0], 'b')
    # fig2 = plt.figure()
    # ax2 = fig2.gca()
    # ax2.set_title("Green")
    # ax2.plot(x[:,0], pdf[:,1], 'r')
    # ax2.plot(x[:,1], pdf[:,1], 'g')
    # ax2.plot(x[:,2], pdf[:,1], 'b')
    # fig3 = plt.figure()
    # ax3 = fig3.gca()
    # ax3.set_title("Yellow")
    # ax3.plot(x[:,0], pdf[:,2], 'r')
    # ax3.plot(x[:,1], pdf[:,2], 'g')
    # ax3.plot(x[:,2], pdf[:,2], 'b')
    # plt.show()

    # Extract PDF for every cluster
    print("pdf before reshaping",pdf.shape)
    pdf_r = pdf[:,0].reshape((shape[0], shape[1]))
    pdf_g = pdf[:,1].reshape((shape[0], shape[1]))
    pdf_y = pdf[:,2].reshape((shape[0], shape[1]))
    pdf = pdf.reshape(shape)

    # segment buoys

    print("pdf after reshaping", pdf.shape)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(shape[0]):
        for j in range(shape[1]):
            max_prob = max(pdf[i][j])
            if max_prob == pdf_r[i][j] and pdf_r[i][j] > 1.5*10**-5:
                seg_o[i][j] = 255
            elif max_prob == pdf_y[i][j] and pdf_y[i][j] > 1*10**-4:
                seg_y[i][j] = 255
            elif max_prob == pdf_g[i][j] and pdf_g[i][j] > 10**-5:
                seg_g[i][j] = 255

            # elif pdf_g[i][j] > pdf_r[i][j] and pdf_g[i][j] > pdf_y[i][j] and pdf_g[i][j] > 10**-5:
            #     seg_image[i][j][1] = 255
                # print(seg_image[i][j])

    img = detect_green(img,seg_g,(0,255,0))
    img = detect_orange(img,seg_o,(0,165,255))
    img = detect_yellow(img,seg_y,(0,255,255))


    # img = detect_orange(img,seg_o)
    # detect_buoys(img,seg_y)

    if not os.path.exists("Data/Output/Frames/"):
        os.makedirs("Data/Output/Frames/")
    cv2.imwrite("Data/Output/Frames/" + name, img)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def detect_green(img,seg,color):

    # ret, threshold = cv2.threshold(seg_g, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)

    # dilation = cv2.dilate(seg_g, kernel, iterations=9)
    closing = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


    _, contours, _= cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours green",len(contours))

    # contours_g = []
    # for con in contours1:
    #     area = cv2.contourArea(con)
    #     if 600 < area < 1000:
    #         contours_g.append(con)

    circles = []

    for con in contours:

        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        print("circularity",circularity)
        if 0.3 < circularity < 1.2:              # perfect circularity 0.68
            circles.append(con)

    print("circles green",len(circles))

    for contour in circles:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y) - 1)
        radius = int(radius) - 1
        # print("radius",radius)
        if radius > 7:                           # perfect buoy radius 10
            cv2.circle(img, center, radius, color, 2)
    return img

def detect_orange(img,seg,color):

    # ret, threshold = cv2.threshold(seg_g, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)

    # dilation = cv2.dilate(seg_g, kernel, iterations=9)
    closing = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


    _, contours, _= cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours orange",len(contours))

    # contours_g = []
    # for con in contours1:
    #     area = cv2.contourArea(con)
    #     if 600 < area < 1000:
    #         contours_g.append(con)

    circles = []

    for con in contours:

        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        # print("circularity",circularity)
        if 0.5 < circularity < 1.2:              # perfect circularity 0.68
            circles.append(con)

    # print("circles orange",len(circles))

    for contour in circles:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y) - 1)
        radius = int(radius) - 1
        print("radius",radius)
        if radius > 7:                           # perfect buoy radius 10
            cv2.circle(img, center, radius, color, 2)
    return img

def detect_yellow(img,seg,color):

    # ret, threshold = cv2.threshold(seg_g, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)

    # dilation = cv2.dilate(seg_g, kernel, iterations=9)
    closing = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


    _, contours, _= cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours yellow",len(contours))

    # contours_g = []
    # for con in contours1:
    #     area = cv2.contourArea(con)
    #     if 600 < area < 1000:
    #         contours_g.append(con)

    circles = []

    for con in contours:

        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        # print("circularity",circularity)
        if 0.5 < circularity < 1.2:              # perfect circularity 0.68
            circles.append(con)

    print("circles yellow",len(circles))

    for contour in circles:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y) - 1)
        radius = int(radius) - 1
        print("radius",radius)
        if radius >= 5:                           # perfect buoy radius 10
            cv2.circle(img, center, radius, color, 2)
    return img


def start_training():
    X = []
    image_paths = glob.glob("Data/Yellow/Extracted/*")
    for image_path in image_paths[:10]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])

    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, None, False)
    X = []
    for image_path in image_paths[10:20]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[20:30]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[30:40]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[40:50]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[50:60]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[60:70]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[70:80]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[80:90]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[90:100]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[100:110]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[110:120]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[120:130]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    X = []
    for image_path in image_paths[130:]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 30, clusters, True)
    f = open("clusters_yellow.txt", "a+")
    for cluster in clusters:
        f.write(str(cluster['mu_k']))
        f.write(str(cluster['cov_k']))
        f.write(str(cluster['pi_k']))
    f.close()
   
    
if __name__ == '__main__':
    # start_training()
    X = []
    # Red
    params = []
    params.append([{'mu_k': [224.24308412, 188.40766512, 135.35937751],
               'cov_k': [[ 513.08028843,  352.57410638,  189.91628892],
                         [ 352.57410638, 1421.76477916,  881.23881318],
                         [ 189.91628892,  881.23881318,  609.7800478 ]],
               'pi_k': [0.14211061]},
              {'mu_k': [243.82040195, 187.27562985, 115.37300701],
               'cov_k': [[  60.89764822, -214.41121252, -129.23936416],
                         [-214.41121252, 1000.80112658,  588.03967894],
                         [-129.23936416,  588.03967894,  419.87880931]],
               'pi_k': [0.68677214]},
              {'mu_k': [254.77915359, 142.38696293,  89.39158317],
               'cov_k': [[ 1.72772990e-01, -3.86259090e-01, -1.29844429e-01],
                         [-3.86259090e-01,  1.45855360e+02,  1.14584737e+02],
                         [-1.29844429e-01,  1.14584737e+02,  1.45921738e+02]],
               'pi_k': [0.17111725]}
    ])
    #Green
    params.append([{'mu_k': [145.85367447, 229.33504252, 130.50776874],
               'cov_k':[[267.68844795, 147.12067433, 133.39614562],
                        [147.12067433, 159.81581508,  49.80439924],
                        [133.39614562,  49.80439924,  94.65929345]],
               'pi_k': [0.29864214]},
              {'mu_k': [197.40240215, 245.68696359, 154.64136134],
               'cov_k':[[403.92980109,   6.12551991, 243.85808718],
                        [  6.12551991,  24.20421411,  29.22713264],
                        [243.85808718,  29.22713264, 214.80791633]],
               'pi_k': [0.24938503]},
              {'mu_k': [107.44334583, 189.26138203, 116.32419582],
               'cov_k': [[155.40240711, 180.9066731,   78.33404419],
                        [180.9066731,  330.25013577, 102.67266542],
                        [ 78.33404419, 102.67266542,  71.72121417]],
               'pi_k': [0.45197282]}])
    # yellow
    params.append([{'mu_k': [212.86290854, 220.68219435, 118.23609258],
              'cov_k':[[1031.79711432,  584.75558749,   36.38668374],
                       [ 584.75558749,  456.76008394,  278.65632245],
                       [  36.38668374,  278.65632245,  858.18023939]],
              'pi_k': [0.39008973]},
             {'mu_k': [229.51226295, 240.6241885,  139.21091162],
              'cov_k':[[  7.38586612,   1.51749218, -42.08377392],
                       [1.51749218,   4.52050088, - 29.73265531],
                       [-42.08377392, - 29.73265531, 772.02085145]],
              'pi_k': [0.55869529]},
             {'mu_k': [82.88271189, 91.51975606, 61.31964543],
              'cov_k': [[2505.82094342, 2687.98151126, 1836.63756781],
                        [2687.98151126, 2944.4162938,  2029.92877086],
                        [1836.63756781, 2029.92877086, 1480.67198803]],
              'pi_k': [0.05121499]}])

    # image = cv2.imread("Data/Green/Test/frame008.png")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "Frames/")
    count =0
    for name in sorted(os.listdir(img_dir)):
        image = cv2.imread(os.path.join(img_dir, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
        create_pdf(params, X, image, image.shape, name)
        count =count +1
        # if count==1:
        #     break