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

W=43054;

%----get labeling p1, p2 and p3--------------------------------------------
p1_centroids=readNPY('GAD67_centroids.npy');
p1_probability=readNPY('GAD67_probability.npy');

p2_centroids=readNPY('GFAP_centroids.npy');
p2_probability=readNPY('GFAP_probability.npy');

p3_centroids=readNPY('Glutaminase_centroids.npy');
p3_probability=readNPY('Glutaminase_probability.npy');

%---retrieve labeling with your favorite classifier------------------------
%must be n_point x n _class probability float matrix
initial_classif_p1 = p1_probability;
initial_classif_p2 = p2_probability;
initial_classif_p3 = p3_probability;

%----build the adjacency graph from ply file-------------------------------
%graph_p1 = build_graph_from_points(p1_centroids,0,10,0,0,4);
%graph_p2 = build_graph_from_points(p2_centroids,size(p1_centroids,1),10,0,0,4);
%graph_p3 = build_graph_from_points(p3_centroids,size(p1_centroids,1)+size(p2_centroids,1),10,0,0,4);

%---retrieve labeling with your favorite classifier------------------------
%must be n_point x n _class probability float matrix
%initial_classif_p1 = p1_probability;
%initial_classif_p2 = p2_probability;
%initial_classif_p3 = p3_probability;

%---All combined ----------------------------------------------------------
%all_data(1,1).x_centroids=p1_centroids(:,1);
%all_data(1,1).name='GAD67';
%all_data(1,1).initial_classif=initial_classif_p1;
%all_data(1,1).graph=graph_p1;
%all_data(1,1).weight=1.5;

%all_data(2,1).x_centroids=p2_centroids(:,1);
%all_data(2,1).name='GFAP';
%all_data(2,1).initial_classif=initial_classif_p2;
%all_data(2,1).graph=graph_p2;
%all_data(2,1).weight=1.0;

%all_data(3,1).x_centroids=p3_centroids(:,1);
%all_data(3,1).name='Glutaminase';
%all_data(3,1).initial_classif=initial_classif_p3;
%all_data(3,1).graph=graph_p3;
%all_data(3,1).weight=0.5;
%--------------------------------------------------------------------------
%--------------- BENCHMARKING ---------------------------------------------
%------------------------- Total Variation --------------------------------
%p_kl_tv = PFDR(all_data, 0.5, 1);
%[dump, l_kl_tv] = max(p_kl_tv,[],1);



for n_neighbor=25:10:25
	for edge_mode=50:50:50
		for dist_type= 2:1:3
			%----build the adjacency graph from ply file-------------------------------
			graph_p1 = build_graph_from_points(p1_centroids,0,n_neighbor,0,edge_mode,dist_type);
			graph_p2 = build_graph_from_points(p2_centroids,size(p1_centroids,1),n_neighbor,0,edge_mode,dist_type);
			graph_p3 = build_graph_from_points(p3_centroids,size(p1_centroids,1)+size(p2_centroids,1),n_neighbor,0,edge_mode,dist_type);
			for w1=1.0:0.5:1.5
				for w2=1.5:0.5:1.5
					for w3=0.5:0.5:1.5
						%---All combined ----------------------------------------------------------
						all_data(1,1).x_centroids=p1_centroids(:,1);
						all_data(1,1).name='GAD67'
						all_data(1,1).initial_classif=initial_classif_p1;
						all_data(1,1).graph=graph_p1;
						all_data(1,1).weight=w1;

						all_data(2,1).x_centroids=p2_centroids(:,1);
						all_data(2,1).name='	GFAP';
						all_data(2,1).initial_classif=initial_classif_p2;
						all_data(2,1).graph=graph_p2;
						all_data(2,1).weight=w2;

						all_data(3,1).x_centroids=p3_centroids(:,1);
						all_data(3,1).name='Glutaminase';
						all_data(3,1).initial_classif=initial_classif_p3;
						all_data(3,1).graph=graph_p3;
						all_data(3,1).weight=w3;
						%--------------------------------------------------------------------------
						%------------------------- Total Variation --------------------------------
						for lambda=0.0:0.5:1.0
							for fidelity=0:1:3
								str_file=strcat('results/labels_',num2str(n_neighbor),'_',num2str(edge_mode),'_',num2str(dist_type),'_',num2str(10*w1),'_',num2str(10*w2),'_',num2str(10*w3),'_',num2str(10*lambda),'_',num2str(fidelity),'.npy')
								if isfile(str_file)
								    disp('File exists')% File exists.
								else
								    % File does not exist.
									disp('File does not exist');
									str_file
									p_kl_tv = PFDR(all_data, lambda, fidelity);
									[dump, l_kl_tv] = max(p_kl_tv,[],1);
									writeNPY(l_kl_tv,str_file);
								end
							end
						end
					end
				end
			end
		end
	end
end	