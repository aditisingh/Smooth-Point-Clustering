function p_regularized = PFDR(all_data, lambda, fidelity)
%Preconditonned Forward Douglas Radchford Algorithm to solve
%TV penalized simplex bound energies
%INPUT
%inital_labeling = classification to regularize
%graph  = the adjacency structure
%fidelity = which fidelity fucntion to use (default = 1)
%	0 : linear
%	1 : quadratic  
%	2 : KL with 0.05 uniform smoothing
%	3 : loglinear with 0.05 uniform smoothing
%lambda      : regularization strength (default = 1)
%benchMarking: if true will return the energy and time for the algorithm
%            : stopped after 1 ... maxIte iteration, starting from zero 
%              each time
%OUTPUT
%p_regularized = the regularized probability
%loic landrieu 2016
%
%When using this method you must cite:
%
%A Note on the Forward-Douglas--Rachford Splitting for Monotone Inclusion
%and Convex Optimization.
%Raguet, H. (2017).
smoothing = 0.05;
if (nargin < 2)
    lambda = 1;
end
if (nargin < 3)
    fidelity = 1;
end

nChannels=numel(all_data);
initial_p={};
x_points={};
graph_sources={};
graph_targets={};
graph_weights={};
fidelity_weights={};

for i=1:nChannels
  initial_p{i}=all_data(i).initial_classif'; %double
  x_points{i}=all_data(i).x_centroids'; %int64
  graph_sources{i}=int32(all_data(i).graph.source); %int32
  graph_targets{i}=int32(all_data(i).graph.target); %int32
  graph_weights{i}=all_data(i).graph.edge_weight*lambda; %single
  fidelity_weights{i}=all_data(i).weight;
end

nClasses  = size(initial_p{1},1);

switch fidelity
    case 0
       p_regularized = PFDR_graph_loss_d1_simplex_mex(initial_p,x_points,0.5, 0 ,...
        graph_sources ,  graph_targets...
        , graph_weights, fidelity_weights, 1, 0.2, 1e-1, 1, 100, 0);
     case 1
       p_regularized = PFDR_graph_loss_d1_simplex_mex(initial_p,x_points,0.5, 1 ,...
        graph_sources ,  graph_targets...
        , graph_weights, fidelity_weights, 1, 0.2, 1e-1, 1, 100, 0);
    case 2
      p_regularized = PFDR_graph_loss_d1_simplex_mex(initial_p,x_points,0.5, smoothing ,...
        graph_sources ,  graph_targets...
        , graph_weights, fidelity_weights, 1, 0.2, 1e-1, 1, 100, 0);
    case 3
       for i=1:nChannels
          %smoothing done
          initial_p{i}=log(initial_p{i}*(1-smoothing  + smoothing/nClasses));
        end
       p_regularized = PFDR_graph_loss_d1_simplex_mex(initial_p, x_points, 0.5, 0,...
        graph_sources ,  graph_targets...
        , graph_weights, fidelity_weights, 1, 0.2, 1e-1, 1, 200, 0);
end
    
