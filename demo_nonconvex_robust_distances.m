% -------------------------------------------------------------------------
% This script tests robustness of a distance matrix structured as follows
% We are given a set of p points in r-dimensions.
% m of these points are anchors = We know the pairwise disttances between
% the anchors
% n of these points are called target nodes = We do not know the distances
% between the target nodes.
% We know all the distances between the target nodes and anchors, but
% they are highly corrupted.
% For this problem, we work with the squared distance matrix D, which
% has rank at most r+2. This means that D is rank at most r+2. 
% The structure of D is as follows:
% D = [E   F
%      F^T G]
% E = (m*m) matrix of anchor-anchor squared distances
% F = (m*n) matrix of anchor-target nodes squared distances
% G = (n*n) matrix of target-target squared distances
% Note that G is not observed
% -------------------------------------------------------------------------
% Load points
outs_cell = pdb2mat("1ubq.pdb");
P = [outs_cell.X ;outs_cell.Y; outs_cell.Z];
% Set up problem parameters
% m = number of anchors
% n = number of target nodes
% p = m+n
% r: embedding dimension
sz_P = size(P);
r = sz_P(1);
p = sz_P(2);
m = 20; 
n = p - m ;
% Create the ground squared distance matrix D
dist = squareform(pdist(P'));
D = dist.*dist;
% Blocks of D
E = D(1:m,1:m);
F = D(1:m,m+1:end);
G = D(m+1:end,m+1:end);
% Add outliers
% We choose a random subset of the squared distances in each column of F
% and corrupt it as follows
% D(i,j) = (1+epsilon*N(0,1)*D(i,j)
% Note N(0,1) = standard gaussian noise
% The above ensures that the relative error, up to the randomnes is
% epsilon
% k = denotes the number of anchor distances that are highly corrupted
epsilon = 0.1;
F_corrupted = F;
for i = 1:n
    % choose k random indices to perturb
    rand_idx = randperm(m);
    rand_idx = rand_idx(1:k);
    F_corrupted(rand_idx,i)= (1+epsilon*randn).*F(rand_idx,i); 
end
% Algorithm for non-convex robust goes here

% Compute relative error in F
error = norm(F-F_estimated,"fro"),norm(F,"fro");
