clear;
addpath rpca\

% Define parameters
m_values = 10:10:60;
alpha_values = 0.1:0.1:0.3;
n_trials = 50; % Number of trials
res_file = "results/res_synthetic_tr50_comp_F_vs_B_d3.txt";

% Initialize results matrices
mn_new = zeros(length(m_values), length(alpha_values));
mn_old = zeros(length(m_values), length(alpha_values));


std_new = zeros(length(m_values), length(alpha_values));
std_old = zeros(length(m_values), length(alpha_values));


% Run trials and store RMSE and standard deviation values
for i = 1:length(m_values)
    for j = 1:length(alpha_values)
        [rmse_new, rmse_old] = run_trial_synthetic(m_values(i), alpha_values(j), n_trials);
        
        mn_new(i, j) = mean(rmse_new);
        mn_old(i, j) = mean(rmse_old);

        std_new(i, j) = std(rmse_new);
        std_old(i, j) = std(rmse_old);
    end
end

% markdown table generation
fid = fopen(res_file, 'w');

fprintf(fid, '### Comparison (num_trials = %d)\n', n_trials);
fprintf(fid, '#### New (RPCA on B)\n');
write_markdown_table(fid, alpha_values, m_values, mn_new, std_new);

fprintf(fid, '#### Old (RPCA on F)\n');
write_markdown_table(fid, alpha_values, m_values, mn_old, std_old);

% Close file
fclose(fid);

%% Trials
function [rmse_new, rmse_old] = run_trial_synthetic(m, alpha, n_trials)
    rmse_new = zeros(n_trials, 1);
    rmse_old = zeros(n_trials, 1);

    for trial = 1:n_trials
        p = 500;  % number of points
        d = 3;    % dimension of the points
        n = p - m;

        % generates points that are more evenly distributed across the given range using Halton sequence   

        hs = haltonset(d, 'Skip', 1e3, 'Leap', 1e2);  % Skip initial values and leap every 100 points to reduce correlation 
        hs = scramble(hs, 'RR2');  % Scramble for more randomness
        P1 = -100 + 200 * net(hs, m); % evenly distributed across the given range using Halton sequence
        P2 = -100+200.*rand(d,n); 
        P = [P1' P2];
        r = d+2;  % rank of the distance matrix

        
        % Ground distance matrix
        dist = squareform(pdist(P'));
        D = dist.*dist;

        % Blocks of D
        % m = round(4*(d+2)*log(p));
        E = D(1:m,1:m);
        F = D(1:m,m+1:end);
        G = D(m+1:end,m+1:end);
        % EF = [E F];


        % RPCA params 
        para.mu        = 1.1*get_mu_kappa(F,r);  
        para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
        para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
        para.trimming  = false;
        para.tol       = 1e-14;
        para.gamma     = 0.9;
        para.max_iter  = 500;

        % sparse noise

        % only F is corrupted
        F_corrupted = get_sparse_noise(F, alpha);
        % D_corrupted = [E F_corrupted; F_corrupted' G];

        % apply SREDG
        X_estimated_new = SREDG_rpcaB(E, F_corrupted, r, @AccAltProj, para);
        X_estimated_old = SREDG(E, F_corrupted, r, @AccAltProj, para);

        % calculate RMSE
        rmse_new(trial) = gram2rmse(X_estimated_new, P, d);
        rmse_old(trial) = gram2rmse(X_estimated_old, P, d);
    end
end


function rmse = gram2rmse(X_estimated, P, d)
    [V, Lam] = eigs(X_estimated, d, 'lm');
    P_estimated = V*sqrt(Lam);
    [rmse, ~, ~] = Compute_RMSE(P',P_estimated);    
end

function F_corrupted = get_sparse_noise(F, alpha)
    m = size(F, 1);
    n = size(F, 2);
    S_supp_idx = randsample(m*n, round(alpha*m*n), false);
    S_range = 1*mean(mean(abs(F)));
    S_temp = 2*S_range*rand(m,n)-S_range; 
    S_true = zeros(m, n);
    S_true(S_supp_idx) = S_temp(S_supp_idx);  
    F_corrupted = F + S_true;
end
