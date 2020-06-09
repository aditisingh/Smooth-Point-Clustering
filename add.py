from PIL import Image
import random
from scipy.spatial import distance
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from skimage.external import tifffile as tiff
from skimage.draw import line_aa
from scipy import ndimage as ndi
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay
from skimage.io import imsave
from skimage import img_as_uint, img_as_bool
import pandas as pd
from skimage.draw import line_aa
from scipy import ndimage as ndi
from skimage import feature
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from matplotlib.pyplot import cm
from itertools import cycle, islice
from sklearn.preprocessing import MinMaxScaler
import os

# os.mkdir('plots_new')

input_path='/project/ece/roysam/aditi/layer_results_maui/datasets/50_plex/'
mask_num=1
# #get image
im_neun = tiff.imread(input_path+'S1_R2C4.tif')
im_mask = Image.open(input_path+'cortex_layers_ver'+str(mask_num)+'.tif')
#
[r,c]=np.where(np.array(im_mask)==True)
im_neun[r,c]=np.max(im_neun)
del im_mask

##python
c1=np.load('../initial_clustering/GAD67_centroids.npy')
p1=np.load('../initial_clustering/GAD67_probability.npy')
l1=np.argmax(p1,axis=1)

l1_new_0=np.load('results/GAD67_labels_new.npy').tolist()
l1_new_0=[int(l-1) for l in l1_new_0[0]]
#plotting all before and after
color = cm.rainbow(np.linspace(0, 1, 16))
np.random.shuffle(color)
colors = np.array(list(islice(cycle(color),int(max(l1_new_0)) + 1)))
fig = plt.figure(figsize=(20,15))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(im_neun,cmap='gray')
plt.scatter(x=c1[:, 0], y=c1[:, 1], s=5, color=colors[l1_new_0])
ax.axis('tight')
ax.axis('off')
plt.savefig('plots_new/GAD67_v1.png')
plt.show()
plt.close()
##python
c1=np.load('initial_clustering/GFAP_centroids.npy')
p1=np.load('initial_clustering/GFAP_probability.npy')
l1=np.argmax(p1,axis=1)

for dist_type in range(0,5+1,1):
	for lamda in range(0,10+1,1):
		for fidelity in range(0,3+1,1):
			result_str=str(int(lamda))+'_'+str(dist_type)+'_'+str(fidelity)
			l1_new_0=np.load('smooth_clustering/results/GFAP_labels_new_new_'+result_str+'.npy').reshape(-1)
			#matlab to python index val conversion
			l1_new_0=[int(l-1) for l in l1_new_0]
			#plotting all before and after
			color = cm.rainbow(np.linspace(0, 1, 16))
			np.random.shuffle(color)
			colors = np.array(list(islice(cycle(color),int(max(l1_new_0)) + 1)))
			fig = plt.figure(figsize=(20,15))
			ax = plt.Axes(fig, [0., 0., 1., 1.])
			ax.set_axis_off()
			fig.add_axes(ax)
			plt.imshow(im_neun,cmap='gray')
			plt.scatter(x=c1[:, 0], y=c1[:, 1], s=5, color=colors[l1_new_0])
			ax.axis('tight')
			ax.axis('off')
			plt.savefig('plots_new/GFAP_final_'+result_str+'.png')
			plt.show()
			plt.close()


##python
c1=np.load('initial_clustering/Glutaminase_centroids.npy')
p1=np.load('initial_clustering/Glutaminase_probability.npy')
l1=np.argmax(p1,axis=1)

for dist_type in range(0,5+1,1):
	for lamda in range(0,10+1,1):
		for fidelity in range(0,3+1,1):
			result_str=str(int(lamda))+'_'+str(dist_type)+'_'+str(fidelity)
			l1_new_0=np.load('smooth_clustering/results/Glutaminase_labels_new_new_'+result_str+'.npy').reshape(-1)
			#matlab to python index val conversion
			l1_new_0=[int(l-1) for l in l1_new_0]
			#plotting all before and after
			color = cm.rainbow(np.linspace(0, 1, 16))
			np.random.shuffle(color)
			colors = np.array(list(islice(cycle(color),int(max(l1_new_0)) + 1)))
			fig = plt.figure(figsize=(20,15))
			ax = plt.Axes(fig, [0., 0., 1., 1.])
			ax.set_axis_off()
			fig.add_axes(ax)
			plt.imshow(im_neun,cmap='gray')
			plt.scatter(x=c1[:, 0], y=c1[:, 1], s=5, color=colors[l1_new_0])
			ax.axis('tight')
			ax.axis('off')
			plt.savefig('plots_new/Glutaminase_final_'+result_str+'.png')
			plt.show()
			plt.close()