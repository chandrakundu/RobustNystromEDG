
% This script is for sensor localization synthetic data. 

%% load data
clear;
addpath E:\projects\edg\RobustNystromEDG\rpca\

p = 100;  % number of points
d = 2;    % dimension of the points
m = 20;
alpha = 0.2; % percentage of outliers
n = p - m;
% P = randn(d, p); 
hs = haltonset(d, 'Skip', 1e3, 'Leap', 1e2);
hs = scramble(hs, 'RR2');
P1 = -100 + 200 * net(hs, m);
P2 = -100+200.*rand(d,n);
P = [P1' P2];

% Gram matrix
X = P'*P; 

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
S_temp = 2*S_range*rand(m,n)-S_range; 
S_true = zeros(m, n);
S_true(S_supp_idx) = S_temp(S_supp_idx);  
F_corrupted = F + S_true;

error = norm(F-F_corrupted,"fro")/norm(F,"fro");
fprintf("Error of F after the corruption: %f\n", error);



%% ROBUST PCA Params
para.mu        = 1.1*get_mu_kappa(F,r);  
para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
para.trimming  = false;
para.tol       = 1e-14;
para.gamma     = 0.9;
para.max_iter  = 500;


%% Apply SREDG
X_estimated = SREDG_dist(E,F_corrupted,r,@AccAltProj, para);

error = norm(X-X_estimated,"fro")/norm(X,"fro");
fprintf("Error of X after the estimation: %f\n", error);


%% Compute estimated points
[V, Lam] = eigs(X_estimated, d, 'lm');
P_estimated = V*sqrt(Lam);


%% Compute RMSE
[rmse, ~, ~] = Compute_RMSE(P',P_estimated)


%% Visualization
plot_points(P',P_estimated,m)
