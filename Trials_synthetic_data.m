clear;
addpath rpca\

% Define parameters
m_values = 10:5:40;
alpha_values = 0.05:0.05:0.3;
n_trials = 50; % Number of trials
res_file = "results/res_synthetic_tr50_mean_with_std.txt";

% Initialize results matrices
rmse_matrix = zeros(length(m_values), length(alpha_values));
std_matrix = zeros(length(m_values), length(alpha_values));

% Run trials and store RMSE and standard deviation values
for i = 1:length(m_values)
    for j = 1:length(alpha_values)
        [rmse, std_dev] = run_trial_synthetic(m_values(i), alpha_values(j), n_trials);
        rmse_matrix(i, j) = rmse; % Store RMSE in matrix
        std_matrix(i, j) = std_dev; % Store standard deviation in matrix
    end
end

% Open file for writing
fid = fopen(res_file, 'w');

% Write Markdown table header
fprintf(fid, '| m \\ alpha |');
for alpha = alpha_values
    fprintf(fid, ' %.2f |', alpha);
end
fprintf(fid, '\n|---|');
fprintf(fid, repmat('---|', 1, length(alpha_values)));

% Write data rows
for i = 1:length(m_values)
    fprintf(fid, '\n| %d |', m_values(i));
    for j = 1:length(alpha_values)
        fprintf(fid, ' %.4f (%.4f) |', rmse_matrix(i, j), std_matrix(i, j));
    end
end

% Close file
fclose(fid);

%% Trials
function [rmse, std_dev] = run_trial_synthetic(m, alpha, n_trials)

    rmses = zeros(n_trials, 1);

    for trial = 1:n_trials
        p = 200;  % number of points
        d = 2;    % dimension of the points
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

        % sparse outliers
        S_supp_idx = randsample(m*n, round(alpha*m*n), false);
        S_range = 1*mean(mean(abs(F)));
        S_temp = 2*S_range*rand(m,n)-S_range; 
        S_true = zeros(m, n);
        S_true(S_supp_idx) = S_temp(S_supp_idx);  
        F_corrupted = F + S_true;

        % RPCA
        para.mu        = 1.1*get_mu_kappa(F,r);  
        para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
        para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
        para.trimming  = false;
        para.tol       = 1e-14;
        para.gamma     = 0.9;
        para.max_iter  = 500;
        [F_estimated, ~] = AccAltProj( F_corrupted, r, para );

        % point estimation after removing noise
        X_estimated = dist2gram_matrix(E, F_estimated, 0.01);

        [V, Lam] = eigs(X_estimated, d, 'lm');
        P_estimated = V*sqrt(Lam);

        % rmse
        [rmses(trial), ~, ~] = Compute_RMSE(P',P_estimated);    
    end

    rmse = mean(rmses);
    std_dev = std(rmses);
end
