
##########################################################
#  Python script template for Question 3 (IAML Level 10)
#  Note that:
#  - You should not change the filename of this file, 'iaml01cw2_q3.py', which is the file name you should use when you submit your code for this question.
#  - You should define the functions shown below in your code.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission
##########################################################

#--- Code for loading the data set and pre-processing --->
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import ward, linkage, dendrogram, single, complete

helpers_path = os.path.join(os.path.dirname(os.getcwd()),'helpers')
sys.path.append(helpers_path)

import iaml01cw2_helpers
from iaml01cw2_helpers import load_CoVoST2

data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
Xtrn,Ytrn,Xtst,Ytst = load_CoVoST2(data_path)

kmeans = KMeans(n_clusters=22, random_state=1)
kmeans.fit(Xtrn)
labels = ['Ar','Ca','Cy','De','En','Es','Et','Fa','Fr','Id','It','Ja','Lv','Mn','Nl','Ru','Sl','Sv','Pt','Ta','Tr','Zh']

#<----

# Q3.1
def iaml01cw2_q3_1():
    print('Inertia', kmeans.inertia_)
    samples_per_cluster = np.zeros(22)
    for i in range(22):
        l = len(kmeans.labels_[kmeans.labels_ == i])
        samples_per_cluster[i] = l
    print(samples_per_cluster)
    
# Q3.2
def iaml01cw2_q3_2():
    mean_l = np.zeros((22,26))
    for i in range(22):
        x = Xtrn[Ytrn == i]
        mean_l[i] = np.mean(x, axis=0)
    mean_l_2d = PCA(n_components=2).fit_transform(mean_l)
    cluster_centres_2d = PCA(n_components=2).fit_transform(kmeans.cluster_centers_)
    plt.figure(figsize=(13,12.5))
    colors = ['blue','green','red','cyan','magenta','yellow','black','orange','brown','tan','maroon','olive','lime','midnightblue','hotpink','darkviolet','goldenrod','teal','palegreen','peru','thistle','dodgerblue']
    target_names = ['Ar','Ca','Cy','De','En','Es','Et','Fa','Fr','Id','It','Ja','Lv','Mn','Nl','Ru','Sl','Sv','Pt','Ta','Tr','Zh']
    lw = 2
    for color, i, target_name in zip(colors, np.arange(0,22,dtype=int), target_names):
        plt.scatter(mean_l_2d[i,0], mean_l_2d[i,1], marker='x', color=color, alpha=.8, lw=lw,
                    label=target_name, cmap=plt.get_cmap('coolwarm'))
        plt.scatter(cluster_centres_2d[i,0], cluster_centres_2d[i,1], marker='^', color=color, alpha=.8, lw=lw,
                    label=target_name, cmap=plt.get_cmap('coolwarm'))
    first_legend = plt.legend(loc='best', shadow=False, facecolor='inherit', scatterpoints=1)
    ax = plt.gca().add_artist(first_legend)
    labels=['mean vector', 'cluster centre']
    markers = [Line2D([0,0],[0,0],color='black', marker='x', linestyle=''), Line2D([0,0],[0,0], color='black', marker='^', linestyle='')]
    plt.legend(markers, labels, numpoints=1)
    plt.grid(True)
    plt.xlim(-4,4)
    plt.ylim(-3,3)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Projection of mean vectors and cluster centres on 2D PCA plane')
    plt.savefig('2d_pca_means_and_centres.png')
    plt.show()

# Q3.3
def iaml01cw2_q3_3():
    mean_l = np.zeros((22,26))
    for i in range(22):
        x = Xtrn[Ytrn == i]
        mean_l[i] = np.mean(x, axis=0)
    Z = ward(mean_l)
    fig = plt.figure(figsize=(12,9))
    dn = dendrogram(Z, orientation='right', leaf_font_size=16, labels=labels)
    plt.title('Dendrogram of hierarchical clustering with Ward\'s linkage on mean vectors', {'fontsize': 24})
    plt.savefig('cluster_dendrogram.png')
    plt.show()

# Q3.4
def iaml01cw2_q3_4():
    cluster_centres_l = np.zeros((22,3,26))
    for i in range(22):
        # isolate language data
        x = Xtrn[Ytrn == i]
        kmeans_l = KMeans(n_clusters=3, random_state=1).fit(x)
        cluster_centres_l[i] = kmeans_l.cluster_centers_
    cluster_centres_c = np.zeros((66,26))
    for i in range(22):
        cluster_centres_c[3*i] = cluster_centres_l[i][0]
        cluster_centres_c[3*i+1] = cluster_centres_l[i][1]
        cluster_centres_c[3*i+2] = cluster_centres_l[i][2]
    Z_ward = ward(cluster_centres_c)
    Z_single = single(cluster_centres_c)
    Z_complete = complete(cluster_centres_c)
    centre_labels = []
    for i in range(22):
        centre_labels.append(labels[i]+', centre 1')
        centre_labels.append(labels[i]+', centre 2')
        centre_labels.append(labels[i]+', centre 3')
    fig = plt.figure(figsize=(25, 30))
    dn = dendrogram(Z_ward, orientation='right', leaf_font_size=16, labels=centre_labels)
    plt.title('Dendrogram of hierarchical clustering with \'ward\' linkage on cluster centres', {'fontsize': 24})
    plt.savefig('cluster_dendrogram_ward.png')
    plt.show()
    fig = plt.figure(figsize=(30, 30))
    dn = dendrogram(Z_single, orientation='right', leaf_font_size=16, labels=centre_labels)
    plt.title('Dendrogram of hierarchical clustering with \'single\' linkage on cluster centres', {'fontsize': 24})
    plt.savefig('cluster_dendrogram_single.png')
    plt.show()    
    fig = plt.figure(figsize=(30, 30))
    dn = dendrogram(Z_complete, orientation='right', leaf_font_size=16, labels=centre_labels)
    plt.title('Dendrogram of hierarchical clustering with \'complete\' linkage on cluster centres', {'fontsize': 24})
    plt.savefig('cluster_dendrogram_complete.png')
    plt.show()

# Q3.5
def iaml01cw2_q3_5():
    Xtrn_0 = Xtrn[Ytrn == 0]
    Xtst_0 = Xtst[Ytst == 0]
    covs = ['full', 'diag']
    Ks = [1,3,5,10,15]
    av_ll = np.zeros((2,2,5))
    for i in range(2):
        for k in range(5):
            gmm = GaussianMixture(n_components=Ks[k],covariance_type=covs[i]).fit(Xtrn_0)
            av_ll[0][i][k] = gmm.score(Xtrn_0)
            av_ll[1][i][k] = gmm.score(Xtst_0)
    print(av_ll)
    av_ll_Xtrn = av_ll[0]
    av_ll_Xtst = av_ll[1]
    K = ['1','3','5','10','15']
    plt.figure(figsize=(12,10))
    plt.scatter(K, av_ll_Xtrn[0], color='black')
    plt.plot(K, av_ll_Xtrn[0], color='green', label='full covariance on training data')
    plt.scatter(K, av_ll_Xtrn[1], color='black')
    plt.plot(K, av_ll_Xtrn[1], color='purple', label='diagonal covariance on training data')
    plt.scatter(K, av_ll_Xtst[0], color='black')
    plt.plot(K, av_ll_Xtst[0], color='red', label='full covariance on testing data')
    plt.scatter(K, av_ll_Xtst[1], color='black')
    plt.plot(K, av_ll_Xtst[1], color='blue', label='diagonal covariance on testing data')
    axes = plt.gca()
    axes.yaxis.grid()
    plt.ylim(10, 30)
    plt.legend(loc='best')
    plt.xlabel('Number of mixture components')
    plt.ylabel('Per-sample average log-likelihood')
    plt.title('Plot of per-sample average log-likelihood versus number of mixture components')
    plt.savefig('av_LL.png')
    plt.show()    

#iaml01cw2_q3_1()   # comment this out when you run the function
#iaml01cw2_q3_2()   # comment this out when you run the function
#iaml01cw2_q3_3()   # comment this out when you run the function
#iaml01cw2_q3_4()   # comment this out when you run the function
#iaml01cw2_q3_5()   # comment this out when you run the function