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
initial_classif_p1 = p1_probability;

for n_neighbor=5:5:100
	for dist_cap=100:500:6000
		for edge_weight_mode=-100:10:100
			graph_p1 = build_graph_from_points(p1_centroids,n_neighbor,dist_cap,edge_weight_mode);
			for fidelity=0:1:3
				for lamda=0.1:0.1:1
					p1_kl_tv = PFDR(initial_classif_p1, graph_p1, lamda, fidelity);
					[dump, l1_kl_tv0] = max(p1_kl_tv,[],1);
					result_str=strcat(int2str(n_neighbor),'_',int2str(dist_cap),'_',int2str(edge_weight_mode),'_',int2str(fidelity),'_',int2str(100*lamda));
					filename=strcat('results/GAD67_labels_new_',result_str,'.npy');
					writeNPY(l1_kl_tv0,filename);
				end
			end
		end
	end
end
