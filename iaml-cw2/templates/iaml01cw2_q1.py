
##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the filename of this file, 'iaml01cw2_q1.py', which is the file name you should use when you submit your code for this question.
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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import seaborn as sns
import sys
import os
from copy import deepcopy
from iaml01cw2_my_helpers import get_sample_num

helpers_path = os.path.join(os.path.dirname(os.getcwd()),'helpers')
sys.path.append(helpers_path)

import iaml01cw2_helpers
from iaml01cw2_helpers import load_mnist, load_FashionMNIST

data_path = os.path.join(os.path.dirname(os.getcwd()))
Xtrn, Ytrn, Xtst, Ytst = load_FashionMNIST(data_path)

Xtrn_orig = deepcopy(Xtrn)
Xtst_orig = deepcopy(Xtst)
Xtrn = Xtrn/255.0
Xtst = Xtst/255.0
Xmean_trn = np.mean(Xtrn, axis=0)
Xmean_tst = np.mean(Xtst, axis=0)
Xtst_nm = Xtst - Xmean_tst
Xtrn_nm = Xtrn - Xmean_trn

Ytrn_cat = np.zeros(len(Ytrn))
for i in range(len(Ytrn)):
    Ytrn_cat[i] = Ytrn[i]+1
Ytst_cat = np.zeros(len(Ytst))
for i in range(len(Ytst)):
    Ytst_cat[i] = Ytst[i]+1
    
pca = PCA().fit(Xtrn_nm)
rec_samples = np.zeros((10,4,784))
#<----

# Q1.1
def iaml01cw2_q1_1():
    print(Xtrn_nm[0,:4])
    print(Xtrn_nm[-1,:4])

    
# Q1.2
def iaml01cw2_q1_2():
    img_samples = np.zeros((50,784))
    for i in range(10):
        Xc = Xtrn[Ytrn_cat == (i+1)] # get the class samples
        mean = np.mean(Xc, axis=0) # compute mean
        dists = np.zeros(len(Xc))
        for j in range(len(Xc)):
            dists[j] = np.linalg.norm(Xc[j]-mean, ord=2)
        Xn = Xc[np.argmin(dists)]
        Xf = Xc[np.argmax(dists)]
        c, d = np.argmin(dists), np.argmax(dists)
        dists[d] = -1
        Xsf = Xc[np.argmax(dists)]
        dists[d] = np.inf
        dists[c] = np.inf
        Xsn = Xc[np.argmin(dists)]
        img_samples[5*i+0] = mean
        img_samples[5*i+1] = Xn
        img_samples[5*i+2] = Xsn
        img_samples[5*i+3] = Xsf
        img_samples[5*i+4] = Xf
    img_samples_grayscale = np.zeros((50,28,28))
    for i in range(50):
        img_samples_grayscale[i] = np.reshape(img_samples[i], (28,28))
    fig, axs = plt.subplots(10,5, figsize=(16,16), sharey=True, sharex=True, subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.5, wspace=0.5))
    fig.suptitle('Displays of mean vector, two closest and two furthest samples per class')
    for i in range(10):
        axs[i,0].imshow(img_samples_grayscale[5*i], cmap=plt.get_cmap('gray_r'))
        axs[i,0].set(xlabel='mean sample of class '+str(i))
        axs[i,1].imshow(img_samples_grayscale[5*i+1], cmap=plt.get_cmap('gray_r'))
        axs[i,1].set(xlabel='class: '+str(i)+',sample number: '+str(get_sample_num(Xtrn, img_samples[5*i+1])))
        axs[i,2].imshow(img_samples_grayscale[5*i+2], cmap=plt.get_cmap('gray_r'))
        axs[i,2].set(xlabel='class: '+str(i)+',sample number: '+str(get_sample_num(Xtrn, img_samples[5*i+2])))
        axs[i,3].imshow(img_samples_grayscale[5*i+3], cmap=plt.get_cmap('gray_r'))
        axs[i,3].set(xlabel='class: '+str(i)+',sample number: '+str(get_sample_num(Xtrn, img_samples[5*i+3])))
        axs[i,4].imshow(img_samples_grayscale[5*i+4], cmap=plt.get_cmap('gray_r'))
        axs[i,4].set(xlabel='class: '+str(i)+',sample number: '+str(get_sample_num(Xtrn, img_samples[5*i+4])))
    plt.savefig('img_vectors.png')
    plt.show()


# Q1.3
def iaml01cw2_q1_3():
    n = len(pca.explained_variance_)
    print(np.sort(pca.explained_variance_)[n-5:n])

# Q1.4
def iaml01cw2_q1_4():
    plt.figure(figsize=(8,8))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color="green", label='curve')
    plt.xlabel('number of principal components')
    plt.ylabel('cumulative explained variance ratio')
    plt.title('Plot of cumulative explained variance ratio versus number of principal components')
    plt.legend(loc='best')
    plt.savefig('Cumvar.png')
    plt.show()

# Q1.5
def iaml01cw2_q1_5():
    fig, axs = plt.subplots(2,5, figsize=(12,12), sharey=True, sharex=True, subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=-0.6, wspace=0.6))
    fig.suptitle('Display of first ten principal components')
    for i in range(5):
        axs[0,i].imshow(pca.components_[i].reshape(28,28), cmap=plt.get_cmap('gray_r'))
        axs[0,i].set(xlabel='image of PC'+ str(i+1))
    for i in range(5):
        axs[1,i].imshow(pca.components_[5+i].reshape(28,28), cmap=plt.get_cmap('gray_r'))
        axs[1,i].set(xlabel='image of PC'+ str(6+i))
    plt.savefig('principal_components_images.png')
    plt.show()

# Q1.6
def iaml01cw2_q1_6():
    class_samples = np.zeros((10,784))
    for i in range(10):
        class_samples[i] = Xtrn_nm[Ytrn_cat == (i+1)][0]
    Ks = [5, 20, 50, 200]
    rmse = np.zeros((10,4))
    for i in range(10):
        x = class_samples[i]
        for k in range(len(Ks)):
            x_ = np.zeros((784,))
            # reconstruct using principal components
            for j in range(Ks[k]):
                x_ = x_ + np.inner(x, pca.components_[j])*pca.components_[j]
            # compute root mean squared error
            rec_samples[i,k] = x_ + Xmean_trn
            rmse[i,k] = np.sqrt(mean_squared_error(x, x_))
    print(rmse)
    
# Q1.7
def iaml01cw2_q1_7():
    fig, axs = plt.subplots(10,4, figsize=(16,16), sharey=True, sharex=True, subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.5, wspace=0.5))
    fig.suptitle('Display of reconstructed images with varying number of principal components')
    for i in range(10):
        axs[i,0].imshow(rec_samples[i,0].reshape(28,28), cmap=plt.get_cmap('gray_r'))
        axs[i,0].set(xlabel='Class '+str(i)+', reconstructed using 5 components')
        axs[i,1].imshow(rec_samples[i,1].reshape(28,28), cmap=plt.get_cmap('gray_r'))
        axs[i,1].set(xlabel='Class '+str(i)+', reconstructed using 20 components')
        axs[i,2].imshow(rec_samples[i,2].reshape(28,28), cmap=plt.get_cmap('gray_r'))
        axs[i,2].set(xlabel='Class '+str(i)+', reconstructed using 50 components')
        axs[i,3].imshow(rec_samples[i,3].reshape(28,28), cmap=plt.get_cmap('gray_r'))
        axs[i,3].set(xlabel='Class '+str(i)+', reconstructed using 200 components')
    plt.savefig('img_reconstructed_vectors.png')
    plt.show()

# Q1.8
def iaml01cw2_q1_8():
    Xtrn_nm_2d = PCA(n_components=2, random_state=1000).fit_transform(Xtrn_nm)
    plt.figure(figsize=(10,10))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'brown', 'white']
    target_names = ['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9']
    lw = 2

    for color, i, target_name in zip(colors, [0,1,2,3,4,5,6,7,8,9], target_names):
        plt.scatter(Xtrn_nm_2d[Ytrn == i, 0], Xtrn_nm_2d[Ytrn == i, 1], marker='x', color=color, alpha=.8, lw=lw,
                    label=target_name, cmap=plt.get_cmap('coolwarm'))
    plt.legend(loc='best',shadow=False, facecolor='lightgray', scatterpoints=1)
    plt.grid(True)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('Projection of fashion-MNIST dataset on 2D PCA plane')
    plt.savefig('2d_pca.png')
    plt.show()

#iaml01cw2_q1_1()   # comment this out when you run the function
#iaml01cw2_q1_2()   # comment this out when you run the function
#iaml01cw2_q1_3()   # comment this out when you run the function
#iaml01cw2_q1_4()   # comment this out when you run the function
#iaml01cw2_q1_5()   # comment this out when you run the function
#iaml01cw2_q1_6()   # comment this out when you run the function
#iaml01cw2_q1_7()   # comment this out when you run the function
#iaml01cw2_q1_8()   # comment this out when you run the function