import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def case2(n = 10, mu = 3, sigma = np.sqrt(5), p = 0.025, rep = 100):
    scaled_crit = ss.norm.ppf(q = 1 - p) * (sigma / np.sqrt(n))
    norm = np.random.normal(loc = mu, scale = sigma, size = (rep, n))
    xbar = norm.mean(1)
    low = xbar - scaled_crit
    up = xbar + scaled_crit
    rem = (mu > low) & (mu < up)
    m = np.c_[xbar, low, up, rem]
    inside = np.sum(m[:, 3])
    per = inside / rep
    desc = "There are " + str(inside) + " confidence intervals that contain the true mean (" + str(mu) + "), that is " + str(per) + " percent of the total CIs"
    return {"Matrix": m, "Decision": desc}

def draw(df):
    plt.style.use('Solarize_Light2')
    plt.show(df.plot(kind = 'box'))

    plt.show(df.mean().plot(kind = 'line'))
    plt.show(df.mean().plot(kind = 'bar'))

def print_sth(df):
    print(df.head())
    print(df.tail())
    print(df.columns)
    print(df.index)
    print(df.T)
    print(df.ix[:0].head())
    print(df.ix[10:20,0:3])
    print(df.drop(df.columns[[1,2]],axis = 1).head())
    print(df.describe())
    print(ss.ttest_1samp(a = df.ix[:,'Abra'],popmean = 15000))
    print(ss.ttest_1samp(a = df,popmean = 15000))

def draw_check():
    im = np.array(Image.open('1.gif'))
    h, w = im.shape
    X = [(h - x, y) for x in range(h) for y in range(w) if im[x][y] < 200]
    X = np.array(X)
    
    n_clusters = 4
    k_means = KMeans(init = 'k-means++' , n_clusters = n_clusters)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    plt.style.use('Solarize_Light2')
    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF3300']
    plt.figure()
    plt.hold(True)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        plt.plot(X[my_members, 1], X[my_members, 0], 'w',
                markerfacecolor=col, marker='.')
        plt.plot(cluster_center[1], cluster_center[0], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    plt.title('KMeans')
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    data_url =            'https://raw.githubusercontent.com/alstat/Analysis-with-Programming/master/2014/Python/Numerical-Descriptions-of-the-Data/data.csv'
    df = pd.read_csv(data_url)
    # print_sth(pd)
    
    # draw(df)
    
    # case2()
    # print(case2()["Decision"])
    draw_check()
