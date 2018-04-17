import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

def load_data(filename):
    with open(filename,'r') as f:
        data = f.readlines()
    feature_list = list([i.split('\t')[0],i.split('\t')[1]] for i in data)
    features = np.array(feature_list)
    print("Shape:%s" % str(features.shape))
    return features
    
color = ['r','g','b','y']

def visuallization(sample , label , center):
    count = 0
    plt.title('GMM')
    for i in center:
        plt.plot(x = i[0],y = i[1],color = color[count])
        pointer = 0
        for j in label:
            if j == count :
                plt.scatter(x = sample[pointer][0],y = sample[pointer][1],color = color[count])
            pointer += 1
        count += 1
    plt.savefig('GMM.png')
    plt.show()

def GMM(sample):
    gmm = GaussianMixture(n_components = 4 , covariance_type = 'tied')
    result = gmm.fit(sample)
    label = gmm.predict(sample)
    center = gmm.means_
    print("center : ",center)
    print("result : ",result)
    print("label : ",label)
    visuallization(sample,label,center)
    
def KMEANS(sample):
    est = KMeans(n_clusters = 4)
    n_clusters = 4
    est.fit(sample)
    k_means_labels = est.labels_
    k_means_cluster_centers = est.cluster_centers_
    inertia = est.inertia_

    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF3300']
    plt.figure()
    plt.hold(True)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        plt.plot(sample[my_members, 1], sample[my_members, 0], 'b',
                markerfacecolor=col, marker='.')
        plt.plot(cluster_center[1], cluster_center[0], 'o', markerfacecolor=col,
                markeredgecolor='w', markersize=6)
    plt.title('KMeans')
    plt.savefig('KMeans.png')
    plt.grid(True)
    plt.show()
    
def D(sample):
    y_pred = DBSCAN().fit_predict(sample)
    plt.title('DBSCAN')
    plt.scatter(sample[:, 0], sample[:, 1], c=y_pred)
    plt.savefig('DBSCAN.png')
    plt.show()
    
if __name__ == '__main__':
    filename = 'Restaurant_Data_Beijing.txt'
    sample = load_data(filename)
    GMM(sample)
    KMEANS(sample)
    D(sample)
