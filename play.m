% create a random matrix of size m x r

n = 1000;
r = 3;
P = rand(n,r);

% create gram matrix
D = P*P';

% create 4 block matrices. top-left: 1 to n and top-right m-n to m
% second is bottom-left 

m = 500;
E = D(1:m,1:m);
F = D(1:m,m+1:end);
G = D(m+1:end,m+1:end);
