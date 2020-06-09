function graph = build_graph_from_points_new_new(points, k, dist_cap,edge_weight_mode,dist_type)
%compute the graph structure
% d0  = average distance between points in the full graph
%INPUT
%points : 2D points 
%single edge_weight_mode = weighting mode of the edges
%   c = 0 : constant weight (default)
%   c > 0 : linear weight w = 1/(d/d0 + c) 
%   c < 0 : exponential weightw = exp(-d/(c * d0))
%OUTPUT
%struct graph =  a structure with the following fields:
%single matrix XYZ : coordinates of each point
%int32 vectors source, target: index of the vertices constituting the
%edges
%single vector edge_weight: the edge of the  
if (nargin < 2)
    k = 10;
end
if (nargin < 3)
    dist_cap = 0;
end
if (nargin < 4)
    edge_weight_mode = 0;
end
if (nargin <5)
	dist_type=0;
end
%dist_type==distance INPUT
% 0 : distance between points
% 1 : distance between points*radial offset between points
% 2 : distance between points*distance diffrence from cortical boundary
% 3 : radial offset between points
% 4 : distance diffrence from cortical boundary
% 5 : radial offset between points*distance diffrence from cortical boundary


graph.XYZ = points;
n_point = size(graph.XYZ,1);

%----adding offset value---------------------------------------------------
outline_img=imread('/project/ece/roysam/aditi/layer_results_maui/datasets/50_plex/cortical_outer_boundary.tif');
[outline_pts_row,outline_pts_column]=find(outline_img>0);
outline_pts=horzcat(outline_pts_column,outline_pts_row); %(x,y)
[closest_pt_on_boundary,cortical_dist] = knnsearch(outline_pts,graph.XYZ,'K',1);


%---compute full adjacency graph-------------------------------------------
[neighbors,distance] = knnsearch(graph.XYZ,graph.XYZ,'K', k+1);

%remove the self edges
neighbors = neighbors(:,2:end);
distance  = distance(:,2:end);
d0     = mean(distance(:));
source      = reshape(repmat(1:n_point, [k 1]), [1 (k * n_point)])';
target      = reshape(neighbors', [1 (k * n_point)])';



%--for each neighbor getting closest_pt_on_boundary and cortical_dist -----
neighbor_cortical_dist=cortical_dist(neighbors);
self_cortical_dist=repmat(cortical_dist(1:n_point),1,k);
cortical_diff=abs(neighbor_cortical_dist-self_cortical_dist);
radial_offset=sqrt(distance.^2-cortical_diff.^2);

if dist_type==0
	distance=distance;
elseif dist_type==1
	distance=distance.*radial_offset;
elseif dist_type==2
	distance=distance.*cortical_diff;
elseif dist_type==3
	distance=radial_offset;
elseif dist_type==4
	distance=cortical_diff;
elseif dist_type==5
	distance=radial_offset.*cortical_diff;
end

%this should be included in distance value
%---edge_weight computation------------------------------------------------
edge_weight = ones(size(distance));
if (edge_weight_mode>0)
    edge_weight = 1./(distance / d0 + edge_weight_mode);
elseif (edge_weight_mode<0)
    edge_weight = exp(distance / (d0 * edge_weight_mode));
end
edge_weight = reshape(edge_weight', [1 (k * n_point)])';
%---pruning----------------------------------------------------------------
pruned = false(size(distance));
if (dist_cap>0)
    pruned = distance > dist_cap * d0;
end
pruned = reshape(pruned, [1 (k * n_point)])';
%---remove self edges and pruned edges-------------------------------------
selfedge = source==target;
to_remove = selfedge + pruned;
source      = source(~to_remove)-1;
target      = target(~to_remove)-1;
edge_weight = edge_weight(~to_remove);
%---symetrizing the graph -------------------------------------------------
double_edges_coord  = [[source;target],[target;source]];
double_edges_weight = [edge_weight;edge_weight];                 
[edges_coord, order] = unique(double_edges_coord, 'rows');
edges_weight = double_edges_weight(order);
%---filling the structure -------------------------------------------------
graph.source      = int32(edges_coord(:,1));
graph.target      = int32(edges_coord(:,2));
graph.edge_weight = single(edges_weight);

