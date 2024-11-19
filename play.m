% create a random matrix of size m x r
clear
n = 10;
r = 2;
P = rand(n,r);
p = size(P,1);
% create gram matrix
D = P*P';

% create 4 block matrices. top-left: 1 to n and top-right m-n to m
% second is bottom-left 

m = 4;
% Randomly select m anchor points
rng(1); % For reproducibility
anchor_indices = randperm(p, m);
target_indices = setdiff(1:p, anchor_indices);

% Blocks of D
E = D(anchor_indices, anchor_indices);
F = D(anchor_indices, target_indices);
G = D(target_indices, target_indices);
