%==========================================================================
%=========== BENCHMARKING THE REGULARIZATION OF SEMANTIC LABELINGS ========
%========================== OF POINT CLOUDS================================
%==========================================================================
%=====================     LOIC LANDRIEU  2017   ==========================
%==========================================================================
%Implementing the methods of the following article
%A structured regularization framework for spatially smoothing semantic
%labelings of 3D point clouds. Landrieu, L., Raguet, H., Vallet, B.,
%Mallet, C., & Weinmann, M. (2017).
%--- dependecies ----------------------------------------------------------
clear all;
addpath('./data')
addpath('./UGM/sub/')
addpath('./UGM/compiled/')
addpath('./UGM/infer/')
addpath('./UGM/decode/')
addpath('./UGM/KPM')
addpath('./GCMex')
addpath('./PFDR_simplex/mex/bin/')
addpath('./cut_pursuit/bin/')
addpath('./func')
addpath('../initial_clustering')

%----get labeling p1, p2 and p3--------------------------------------------
p1_centroids=readNPY('GAD67_centroids.npy');
p1_probability=readNPY('GAD67_probability_unsymm.npy');

p2_centroids=readNPY('GFAP_centroids.npy');
p2_probability=readNPY('GFAP_probability_unsymm.npy');

p3_centroids=readNPY('Glutaminase_centroids.npy');
p3_probability=readNPY('Glutaminase_probability_unsymm.npy');

%----build the adjacency graph from ply file-------------------------------

p1_probability=p3_probability;
p1_centroids=p3_centroids;

%---retrieve labeling with your favorite classifier------------------------
%must be n_point x n _class probability float matrix
initial_classif_p1 = p1_probability;

%----build the adjacency graph from ply file-------------------------------

for n_neighbor=20:5:20
	for dist_type= 3:1:3
		for edge_mode=-50:100:50
			graph_p1 = build_graph_from_points_new_new(p1_centroids,n_neighbor,0,edge_mode,dist_type);
			for fidelity=1:1:3
				for lambda=0.5:0.5:1
				%---Total Variation--------------------------------------------------------
				p1_kl_tv = PFDR(initial_classif_p1, graph_p1, lambda, fidelity);
				[dump, l1_kl_tv] = max(p1_kl_tv,[],1);
				result_str=strcat(num2str(n_neighbor),'_',num2str(dist_type),'_',num2str(edge_mode),'_',num2str(fidelity),'_',num2str(lambda*10),'_')
				filename=strcat('results/Glutaminase_cortical_unsymm_',result_str,'.npy');
				writeNPY(l1_kl_tv,filename);
				end
			end
		end
	end
end


p1_probability=p2_probability;
p1_centroids=p2_centroids;

%---retrieve labeling with your favorite classifier------------------------
%must be n_point x n _class probability float matrix
initial_classif_p1 = p1_probability;

for n_neighbor=25:5:25
	for dist_type= 5:1:5
		for edge_mode=50:50:50
			graph_p1 = build_graph_from_points_new_new(p1_centroids,n_neighbor,0,edge_mode,dist_type);
			for fidelity=0:1:3
				for lambda=0:0.5:1
				%---Total Variation--------------------------------------------------------
				p1_kl_tv = PFDR(initial_classif_p1, graph_p1, lambda, fidelity);
				[dump, l1_kl_tv] = max(p1_kl_tv,[],1);
				result_str=strcat(num2str(n_neighbor),'_',num2str(dist_type),'_',num2str(edge_mode),'_',num2str(fidelity),'_',num2str(lambda*10),'_')
				filename=strcat('results/GFAP_cortical_',result_str,'.npy');
				writeNPY(l1_kl_tv,filename);
				end
			end
		end
	end
end




%----build the adjacency graph from ply file-------------------------------
graph_p1 = build_graph_from_points_new_new(p1_centroids,10,1500,-25,2);
fidelity=3;
dist_type=2;
for lambda=0:0.1:1
	%---Total Variation--------------------------------------------------------
	p1_kl_tv = PFDR(initial_classif_p1, graph_p1, lambda, fidelity);
	[dump, l1_kl_tv] = max(p1_kl_tv,[],1);
	result_str=strcat(int2str(lambda*10),'_',int2str(dist_type),'_',int2str(fidelity))
	filename=strcat('results/GAD67_labels_new_new_',result_str,'.npy');
	writeNPY(l1_kl_tv,filename);
end
	

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

os.mkdir('plots_new')

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
c1=np.load('initial_clustering/GAD67_centroids.npy')
p1=np.load('initial_clustering/GAD67_probability_new.npy')
l1=np.argmax(p1,axis=1)

files=os.listdir('smooth_clustering/results')
file_needed=[f for f in files if f.find('GAD67_cortical')==0]

for f in file_needed:
	l1_new_0=np.load('smooth_clustering/results/'+f).reshape(-1)
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
	plt.savefig('plots_new/'+f[:-4]+'.png')
	plt.show()
	plt.close()

for dist_type in range(0,4+2,2):
	for lamda in range(0,10+1,1):
		for fidelity in range(0,3+1,1):
			result_str=str(int(lamda))+'_'+str(dist_type)+'_'+str(fidelity)
			l1_new_0=np.load('smooth_clustering/results/GAD67_labels_new_new_'+result_str+'.npy').reshape(-1)
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
			plt.savefig('plots_new/GAD67_final_'+result_str+'.png')
			plt.show()
			plt.close()


fidelity=3
dist_type=2

for lamda in range(0,10+1,1):
	result_str=str(int(lamda))+'_'+str(dist_type)+'_'+str(fidelity)
	l1_new_0=np.load('smooth_clustering/results/GAD67_labels_new_new_'+result_str+'.npy').reshape(-1)
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
	plt.savefig('plots_new/GAD67_final_'+result_str+'.png')
	plt.show()
	plt.close()