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
%% Load points
clear;
addpath rpca\
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

X = P'*P; % Gram matrix
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
epsilon = 0.3;
r = r + 2;

alpha = 0.2;


%% non sparse noise
% k = round(alpha*m);
% for i = 1:n
%     % choose k random indices to perturb   
%     rng(3);
%     rand_idx = randperm(m);
%     rand_idx = rand_idx(1:k);
%     F_corrupted(rand_idx,i)= (1+epsilon*randn).*F(rand_idx,i); 
% end
% error = norm(F-F_corrupted,"fro")/norm(F,"fro");
% fprintf("Error of F after the corruption: %f\n", error);

%% sparse noise 
S_supp_idx = randsample(m*n, round(alpha*m*n), false);
S_range = 1*mean(mean(abs(F)));
S_temp = 2*S_range*rand(m,n)-S_range; 
S_true = zeros(m, n);
S_true(S_supp_idx) = S_temp(S_supp_idx);  
F_corrupted = F + S_true;

error = norm(F-F_corrupted,"fro")/norm(F,"fro");
fprintf("Error of F after the corruption: %f\n", error);


%% ACCALTPROJ
para.mu        = 1.1*get_mu_kappa(F,r);  
para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(1*sqrt(m*n));
para.trimming  = false;
para.tol       = 1e-14;
para.gamma     = 0.9;
para.max_iter  = 200;
[F_estimated, ~] = AccAltProj( F_corrupted, r, para );

error = norm(F-F_estimated,"fro")/norm(F,"fro");
fprintf("Error of F after RPCA: %f\n", error);

%% Gram matrix estimation and point estimation after removing noise

X_estimated = dist2gram_matrix(E, F_estimated, 0.1);

error = norm(X-X_estimated,"fro")/norm(X,"fro");
fprintf("Error of X after the estimation: %f\n", error);

[V, Lam] = eigs(X_estimated, r, 'lm');
P_estimated = V*sqrt(Lam);


%% Gram matrix and point estimation no noise case

X_estimated0 = dist2gram_matrix(E, F, 0.1);

error = norm(X-X_estimated0,"fro")/norm(X,"fro");
fprintf("Error of X after the estimation (No Noise): %f\n", error);

[V, Lam] = eigs(X_estimated0, r, 'lm');
P_estimated0 = V*sqrt(Lam);


%% save to pdb 
outs_cell.X = P_estimated0(:,1)';
outs_cell.Y = P_estimated0(:,2)';
outs_cell.Z = P_estimated0(:,3)';
outs_cell.outfile = "1ubq_noise_estimated.pdb";

mat2pdb(outs_cell);
