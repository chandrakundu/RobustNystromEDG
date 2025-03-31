
% This script is for sensor localization synthetic data. 

%% load data
clear; clc;

p = 100;  % number of points
d = 3;    % dimension of the points
m = 35;
alpha = 0.1; % percentage of outliers
n = p - m;
% P = randn(d, p); 
hs = haltonset(d, 'Skip', 1e3, 'Leap', 1e2);
hs = scramble(hs, 'RR2');
P1 = -100 + 200 * net(hs, m);
rng(6);
P2 = -100+200.*rand(d,n);
P = [P1' P2];  % X_star
P = P - mean(P,2);

% Gram matrix
X = P'*P; % L_star

% Ground distance matrix
dist = squareform(pdist(P'));
D = dist.*dist;

% Blocks of D
% m = round(4*(d+2)*log(p));

E = D(1:m,1:m);
F = D(1:m,m+1:end);
G = D(m+1:end,m+1:end);

% Add outliers
r = d + 2; % rank of the distance matrix


%% sparse noise
S_supp_idx = randsample(m*n, round(alpha*m*n), false);
S_range = 1*mean(mean(abs(F)));
% S_range = max(F(:));
S_temp = 2*S_range*rand(m,n)-S_range; 
S_true = zeros(m, n);
S_true(S_supp_idx) = S_temp(S_supp_idx);  
F_corrupted = F + S_true;

% error = norm(F-F_corrupted,"fro")/norm(F,"fro");
% fprintf("Error of F after the corruption: %f\n", error);



%% Apply SREDG
k = m;
known_row_F = F(k,:);  
tol = 1e-18;
zeta0 = 1 * max(F(:));
gamma = 0.9;
maxIter = 100;
show_res = 1;
[X_estimated, ~] = AAP_SREDG(E, F_corrupted, X, P, d, known_row_F, k, zeta0, gamma, maxIter, show_res);


%% Compute estimated points
[V, Lam] = eigs(X_estimated, d, 'lm');
P_estimated = V*sqrt(Lam);


%% Compute RMSE
[rmse, ~, ~] = Compute_RMSE(P',P_estimated);


%% Visualization
% plot_points(P',P_estimated,m)
