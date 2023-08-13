
##########################################################
#  Python script template for Question 2 (IAML Level 10)
#  Note that
#  - You should not change the filename of this file, 'iaml01cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should define the functions shown below in your code.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission
##########################################################

#--- Code for loading the data set and pre-processing --->
import os
import gzip
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate
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

model_lr = LogisticRegression().fit(Xtrn_nm, Ytrn)
model_svc = SVC(kernel='rbf', C=1.0, gamma='auto').fit(Xtrn_nm, Ytrn)

pca = PCA().fit(Xtrn_nm)
V = pca.components_
[sigma1, sigma2] = np.sqrt(pca.explained_variance_)[:2]
xs = np.linspace(-5*sigma1, 5*sigma1, num=100)
ys = np.linspace(-5*sigma2, 5*sigma2, num=100)
xx, yy = np.meshgrid(xs, ys)
z_xy = np.c_[xx.ravel(), yy.ravel()]
z_orig = np.zeros((len(xx)*len(xx), 784))
for i in range(len(xx)*len(xx)):
    z_orig[i][:2] = z_xy[i]
    z_orig[i][2:] = np.zeros((1,782))
z_xy_orig = z_orig.dot(V)

#<----

# Q2.1
def iaml01cw2_q2_1():
    y_pred_lr = model_lr.predict(Xtst_nm)
    print('Accuracy score', accuracy_score(Ytst, y_pred_lr))
    print(confusion_matrix(Ytst, y_pred_lr))

# Q2.2
def iaml01cw2_q2_2():
    y_pred_svc = model_svc.predict(Xtst_nm)
    print(model_svc.score(Xtst_nm, Ytst))
    print(confusion_matrix(Ytst, y_pred_svc))

# Q2.3
def iaml01cw2_q2_3():
    Z = model_lr.predict(z_xy_orig)
    # Put the result into a color plot
    fig, ax = plt.subplots(figsize=(10,8))
    cmap = plt.cm.coolwarm
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(0,10)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('Decision boundaries for multiclass logistic regression on 2D PCA plane')
    cs = ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cmap, norm=norm)
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    ax2.set_ylabel('Classes', size=12)
    plt.savefig('lr_2d_plane.png')
    plt.show()

# Q2.4
def iaml01cw2_q2_4():
    Z_svm = model_svc.predict(z_xy_orig)
    fig, ax = plt.subplots(figsize=(10,8))
    cmap = plt.cm.coolwarm
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(0,10)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('Decision boundaries for multiclass SVM classification on 2D PCA plane')
    cs = ax.contourf(xx, yy, Z_svm.reshape(xx.shape), cmap=cmap, norm=norm)
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    ax2.set_ylabel('Classes', size=12)
    plt.savefig('svc_2d_plane.png')
    plt.show()    

# Q2.5
def iaml01cw2_q2_5():
    Xsmall = np.zeros((10000, 784))
    Ysmall = np.zeros((10000,), dtype=int)
    # allocate 1000 samples per class
    for i in range(10):
        Xsmall[i*1000:(i+1)*1000, :] = (Xtrn_nm[Ytrn == i])[:1000, :]
        Ysmall[i*1000:(i+1)*1000] = i
    C_range = np.logspace(-2, 3, num=10)
    C_range
    mean_cv_scores = np.zeros((10,))
    for i in range(10):
        model_ = SVC(kernel='rbf', C=C_range[i], gamma='auto')
        cv_results = cross_validate(model_, Xsmall, Ysmall, cv=3, scoring='accuracy')
        mean_cv_scores[i] = np.mean(cv_results['test_score'])    
    print(np.max(mean_cv_scores))
    print(C_range[np.argmax(mean_cv_scores)])
    fig, axs = plt.subplots(figsize=(10,7))
    plt.scatter(C_range, mean_cv_scores, color='black', label='point')
    plt.plot(C_range, mean_cv_scores, color='green', label='curve')
    axs.set_xscale('log')
    axs.set_xbound(lower=10**(-3), upper=10**4)
    plt.xlabel('regularisation parameter')
    plt.ylabel('mean cross-validated classification accuracy')
    plt.title('Plot of mean cross-validated classification accuracy versus regularisation parameter')
    plt.legend(loc='best')
    plt.savefig('svc_cv_plot.png')
    plt.show()
    
# Q2.6 
def iaml01cw2_q2_6():
    c_max = 21.544346900318846
    svc_max = SVC(kernel='rbf', C=c_max, gamma='auto').fit(Xtrn_nm, Ytrn)
    y_pred_tr_svc_max = svc_max.predict(Xtrn_nm)
    y_pred_tst_svc_max = svc_max.predict(Xtst_nm)
    print(accuracy_score(Ytrn, y_pred_tr_svc_max))
    print(accuracy_score(Ytst, y_pred_tst_svc_max))
    
iaml01cw2_q2_1()   # comment this out when you run the function
iaml01cw2_q2_2()   # comment this out when you run the function
iaml01cw2_q2_3()   # comment this out when you run the function
iaml01cw2_q2_4()   # comment this out when you run the function
iaml01cw2_q2_5()   # comment this out when you run the function
iaml01cw2_q2_6()   # comment this out when you run the function