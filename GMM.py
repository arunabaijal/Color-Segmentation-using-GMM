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


def train_gmm(X, n_clusters, n_epochs):
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

def create_pdf(params, X, shape):
    seg_image = np.zeros(((shape[1], shape[0])))
    print(seg_image.shape)
    pdf = np.zeros((len(X),3))
    # x = np.array([np.arange(0,256,1), X[:,1], X[:,2]]).transpose()
    x = np.array(X, dtype='uint8')
    for i, clusters in enumerate(params):
        for cluster in clusters:
            y_r = multivariate_normal.pdf(x, cluster['mu_k'], cluster['cov_k'])
            pdf[:,i] = pdf[:,i] + cluster['pi_k'] * y_r
    # plt.plot(x, pdf)
    # plt.show()
    print(pdf.shape)
    pdf_r = pdf[:,0].reshape((shape[1], shape[0]))
    pdf_g = pdf[:,1].reshape((shape[1], shape[0]))
    pdf_y = pdf[:,2].reshape((shape[1], shape[0]))
    print(pdf.shape)
    # max = np.max(pdf)
    # pdf = pdf*255/max
    for i in range(shape[1]):
        for j in range(shape[0]):
            if pdf_g[i][j] > pdf_r[i][j] and pdf_g[i][j] > pdf_y[i][j] and pdf_g[i][j] > 1*10**-5:
                seg_image[i][j] = 255
    cv2.imshow('segmented', seg_image.transpose())
    cv2.waitKey(0)
    

if __name__ == '__main__':
    X = []
    # for image_path in glob.glob("Data/Yellow/Extracted/*")[:20]:
    #     image = cv2.imread(image_path)
    #     image = cv2.resize(image, (30, 30))
    image = cv2.imread("Data/Green/Test/frame008.png")
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])
    params = []
    params.append([{'mu_k': [252.16152635, 152.40805137,  93.57734565],
              'cov_k': [[  9.26513121, -37.31779491, -21.68779137],
                         [-37.31779491, 387.94743064, 228.53106602],
                         [-21.68779137, 228.53106602, 189.12188744]],
              'pi_k': [0.59897148]},
    {'mu_k': [230.37233525, 171.62158315, 121.0135146],
     'cov_k': [[ 596.22083251,  235.6915169,    62.65293133],
               [ 235.6915169,  1108.6769714,   706.83031066],
               [  62.65293133,  706.83031066,  537.28323765]],
     'pi_k': [0.08908288]},
      {'mu_k': [237.03092141, 214.56417343, 131.05491014],
       'cov_k': [[  40.82748049, -112.97472246, -101.0973088 ],
                 [-112.97472246,  447.85877011,  382.74532092],
                 [-101.0973088,   382.74532092,  429.67233197]],
       'pi_k': [0.31194564]}
    ])
    #Green
    params.append([{'mu_k': [198.52851258, 245.92232003, 156.95888053],
               'cov_k':[[405.95235374,  14.57167719, 256.60293682],
                        [ 14.57167719,  27.41013544,  31.93758814],
                        [256.60293682,  31.93758814, 234.29463999]],
               'pi_k': [0.24017603]},
              {'mu_k': [106.67782568, 186.55604151, 116.23520101],
               'cov_k':[[130.62914369, 156.4074056,   71.10019976],
                        [156.4074056,  311.47821003,  95.22541093],
                        [ 71.10019976,  95.22541093,  67.39115444]],
               'pi_k': [0.41947481]},
              {'mu_k': [143.11443463, 227.27641259, 129.40975209],
               'cov_k': [[297.74849462, 192.95405216, 136.55362279],
                         [192.95405216, 195.68659985,  71.51763291],
                         [136.55362279,  71.51763291,  88.74814044]],
               'pi_k': [0.34034916]}])
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
    create_pdf(params, X, image.shape)

def start_training():
    X = []
    for image_path in glob.glob("Data/Red/Extracted/*")[:20]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])

    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 50)
    f = open("clusters_red.txt", "a+")
    for cluster in clusters:
        f.write(str(cluster['mu_k']))
        f.write(str(cluster['cov_k']))
        f.write(str(cluster['pi_k']))
    f.close()
    # create_pdf(clusters, X)
    X = []
    for image_path in glob.glob("Data/Green/Extracted/*")[:20]:
        image = cv2.imread(image_path)
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])

    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 50)
    f = open("clusters_green.txt", "a+")
    for cluster in clusters:
        f.write(str(cluster['mu_k']))
        f.write(str(cluster['cov_k']))
        f.write(str(cluster['pi_k']))
    f.close()
    create_pdf(clusters, X)
    X = []
    for image_path in glob.glob("Data/Yellow/Extracted/*")[:20]:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (30,30))
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if [image[j][i][2], image[j][i][1], image[j][i][0]] != [0,0,0]:
                    X.append([int(image[j][i][2]), int(image[j][i][1]), int(image[j][i][0])])

    clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(np.array(X), 3, 50)
    f = open("clusters_yellow.txt", "a+")
    for cluster in clusters:
        f.write(str(cluster['mu_k']))
        f.write(str(cluster['cov_k']))
        f.write(str(cluster['pi_k']))
    f.close()
    # create_pdf(clusters, X)
    # print(X)