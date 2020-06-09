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
p1_probability=readNPY('GAD67_probability.npy');

p2_centroids=readNPY('GFAP_centroids.npy');
p2_probability=readNPY('GFAP_probability.npy');

p3_centroids=readNPY('Glutaminase_centroids.npy');
p3_probability=readNPY('Glutaminase_probability.npy');

%----build the adjacency graph from ply file-------------------------------
graph_p1 = build_graph_from_points(p1_centroids,10,0,0);
graph_p2 = build_graph_from_points(p2_centroids,10,0,0);
graph_p3 = build_graph_from_points(p3_centroids,10,0,0);

%---retrieve labeling with your favorite classifier------------------------
%must be n_point x n _class probability float matrix
initial_classif_p1 = p1_probability;
initial_classif_p2 = p2_probability;
initial_classif_p3 = p3_probability;

%--------------------------------------------------------------------------
%--------------- BENCHMARKING ---------------------------------------------
%--------------------------------------------------------------------------

%---Total Variation--------------------------------------------------------
p1_kl_tv = PFDR(initial_classif_p1, graph_p1, 0.5, 2);
[dump, l1_kl_tv] = max(p1_kl_tv,[],1);

p2_kl_tv = PFDR(initial_classif_p2, graph_p2, 0.5, 2);
[dump, l2_kl_tv] = max(p2_kl_tv,[],1);

p3_kl_tv = PFDR(initial_classif_p3, graph_p3, 0.5, 2);
[dump, l3_kl_tv] = max(p3_kl_tv,[],1);

writeNPY(l1_kl_tv,'results/GAD67_labels_new.npy')
writeNPY(l2_kl_tv,'results/GFAP_labels_new.npy')
writeNPY(l3_kl_tv,'results/Glutaminase_labels_new.npy')

%--------------- COMBINE ALL ----------------------------------------------
all_centroids=[p1_centroids;p2_centroids;p3_centroids];

[neighbors1,distance1] = knnsearch(p3_centroids,p1_centroids,'K', 10+1);
[neighbors2,distance2] = knnsearch(p3_centroids,p2_centroids,'K', 10+1);

%remove the self edges
neighbors1 = neighbors1(:,2:end);
distance1  = distance1(:,2:end);
neighbors2 = neighbors2(:,2:end);
distance2  = distance2(:,2:end);

%for all points in p1, get probability
p13_probability=p3_probability(neighbors1,:);
p23_probability=p3_probability(neighbors2,:);

p13_probability=reshape(p13_probability,[],10,6);
p23_probability=reshape(p23_probability,[],10,6);

mean_p1=mean(p13_probability,2);
mean_p2=mean(p23_probability,2);

p1_probability_new=reshape(mean_p1,[],6);
p2_probability_new=reshape(mean_p2,[],6);

all_probability=[p1_probability_new;p2_probability_new;p3_probability];
initial_classif=all_probability;

for n_neighbor=5:5:25
	for edge_mode=-50:50:50
		for dist_type= 0:1:5
			graph_full = build_graph_from_points_new_new(all_centroids,n_neighbor,0,edge_mode,dist_type);
			for fidelity=0:1:3
				p_kl_tv = PFDR(initial_classif, graph_full, 0.5, fidelity);
				[dump, l_kl_tv] = max(p_kl_tv,[],1);
				str_file=strcat('results/all_labels_',num2str(n_neighbor),'_',num2str(edge_mode),'_',num2str(dist_type),'_',num2str(fidelity))
				writeNPY(l_kl_tv,strcat(str_file,'.npy'));
			end
		end
	end
end