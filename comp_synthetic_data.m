clear;
addpath rpca\

% Define parameters
m_values = 10:10:60;
alpha_values = 0.1:0.1:0.3;
n_trials = 50; % Number of trials
res_file = "draft_res/res_synthetic_tr50_comp_all.txt";

% Initialize results matrices
mn_F_dist = zeros(length(m_values), length(alpha_values));
mn_F_gram = zeros(length(m_values), length(alpha_values));
mn_F_fullD = zeros(length(m_values), length(alpha_values));
mn_EF_dist = zeros(length(m_values), length(alpha_values));
mn_EF_gram = zeros(length(m_values), length(alpha_values));
mn_EF_fullD = zeros(length(m_values), length(alpha_values));
mn_D = zeros(length(m_values), length(alpha_values));

std_F_dist = zeros(length(m_values), length(alpha_values));
std_F_gram = zeros(length(m_values), length(alpha_values));
std_F_fullD = zeros(length(m_values), length(alpha_values));
std_EF_dist = zeros(length(m_values), length(alpha_values));
std_EF_gram = zeros(length(m_values), length(alpha_values));
std_EF_fullD = zeros(length(m_values), length(alpha_values));
std_D = zeros(length(m_values), length(alpha_values));

% Run trials and store RMSE and standard deviation values
for i = 1:length(m_values)
    for j = 1:length(alpha_values)
        [rmse_F_dist, rmse_F_gram, rmse_F_fullD, rmse_EF_dist, rmse_EF_gram, rmse_EF_fullD, rmse_D] = run_trial_synthetic(m_values(i), alpha_values(j), n_trials);
        
        mn_F_dist(i, j) = mean(rmse_F_dist);
        mn_F_gram(i, j) = mean(rmse_F_gram);
        mn_F_fullD(i, j) = mean(rmse_F_fullD);
        mn_EF_dist(i, j) = mean(rmse_EF_dist);
        mn_EF_gram(i, j) = mean(rmse_EF_gram);
        mn_EF_fullD(i, j) = mean(rmse_EF_fullD);
        mn_D(i, j) = mean(rmse_D);

        std_F_dist(i, j) = std(rmse_F_dist);
        std_F_gram(i, j) = std(rmse_F_gram);
        std_F_fullD(i, j) = std(rmse_F_fullD);
        std_EF_dist(i, j) = std(rmse_EF_dist);
        std_EF_gram(i, j) = std(rmse_EF_gram);
        std_EF_fullD(i, j) = std(rmse_EF_fullD);
        std_D(i, j) = std(rmse_D);
    end
end

% markdown table generation
fid = fopen(res_file, 'w');

fprintf(fid, '### Comparison (num_trials = %d)\n', n_trials);
fprintf(fid, '#### Noise only on F and Nyström on distance\n');
write_markdown_table(fid, alpha_values, m_values, mn_F_dist, std_F_dist);

fprintf(fid, '#### Noise only on F and Nyström on gram\n');
write_markdown_table(fid, alpha_values, m_values, mn_F_gram, std_F_gram);

fprintf(fid, '#### Noise only on F and RPCA on D\n');
write_markdown_table(fid, alpha_values, m_values, mn_F_fullD, std_F_fullD);

fprintf(fid, '#### Noise on both E and F and Nyström on distance\n');
write_markdown_table(fid, alpha_values, m_values, mn_EF_dist, std_EF_dist);

fprintf(fid, '#### Noise on both E and F and Nyström on gram\n');
write_markdown_table(fid, alpha_values, m_values, mn_EF_gram, std_EF_gram);

fprintf(fid, '#### Noise on both E and F and RPCA on D\n');
write_markdown_table(fid, alpha_values, m_values, mn_EF_fullD, std_EF_fullD);

fprintf(fid, '#### Noise in E, F and G; RPCA on D\n');
write_markdown_table(fid, alpha_values, m_values, mn_D, std_D);



% Close file
fclose(fid);

%% Trials
function [rmse_F_dist, rmse_F_gram, rmse_F_fullD, rmse_EF_dist, rmse_EF_gram, rmse_EF_fullD, rmse_D] = run_trial_synthetic(m, alpha, n_trials)
    rmse_F_dist = zeros(n_trials, 1);
    rmse_F_gram = zeros(n_trials, 1);
    rmse_F_fullD = zeros(n_trials, 1);
    rmse_EF_dist = zeros(n_trials, 1);
    rmse_EF_gram = zeros(n_trials, 1);
    rmse_EF_fullD = zeros(n_trials, 1);
    rmse_D = zeros(n_trials, 1);

    for trial = 1:n_trials
        p = 500;  % number of points
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
        EF = [E F];


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
        F_F_corrupted = get_sparse_noise(F, alpha);
        D_F_corrupted = [E F_F_corrupted; F_F_corrupted' G];

        % both E and F is corrupted
        EF_corrupted = get_sparse_noise(EF, alpha);
        E_EF_corrupted = EF_corrupted(1:m,1:m);
        F_EF_corrupted = EF_corrupted(1:m,m+1:end);
        D_EF_corrupted = [E_EF_corrupted F_EF_corrupted; F_EF_corrupted' G];

        % full D is corrupted
        E_D_corrupted = get_sparse_noise(E, alpha);
        F_D_corrupted = get_sparse_noise(F, alpha);
        G_D_corrupted = get_sparse_noise(G, alpha);
        D_D_corrupted = [E_D_corrupted F_D_corrupted; F_D_corrupted' G_D_corrupted];

        % RPCA on F when only F is corrupted
        [F_F_estimated, ~] = AccAltProj( F_F_corrupted, r, para );

        % RPCA on D when only F is corrupted
        [D_F_estimated, ~] = AccAltProj( D_F_corrupted, r, para );
        
        % RPCA on EF when both E and F are corrupted
        [EF_estimated, ~] = AccAltProj( EF_corrupted, r, para );
        E_EF_Estimated = EF_estimated(1:m,1:m);
        F_EF_Estimated = EF_estimated(1:m,m+1:end);

        % RPCA on D when both E and F are corrupted
        [D_EF_estimated, ~] = AccAltProj( D_EF_corrupted, r, para );

        % RPCA on D when only D is corrupted
        [D_D_estimated, ~] = AccAltProj( D_D_corrupted, r, para );


        % === rmse of point estimation after removing noise ===

        % only F is corrupted and nyström on distance
        X_estimated_F_dist = dist2gram(E, F_F_estimated, "dist");
        rmse_F_dist(trial) = gram2rmse(X_estimated_F_dist, P, d);

        % only F is corrupted and nyström on gram
        X_estimated_F_gram = dist2gram(E, F_F_estimated, "gram");
        rmse_F_gram(trial) = gram2rmse(X_estimated_F_gram, P, d);

        % only F is corrupted and nyström on none
        X_estimated_F_fullD = dist2gram(D_F_estimated);
        rmse_F_fullD(trial) = gram2rmse(X_estimated_F_fullD, P, d);

        % both E and F are corrupted and nyström on distance
        X_estimated_EF_dist = dist2gram(E_EF_Estimated, F_EF_Estimated, "dist");
        rmse_EF_dist(trial) = gram2rmse(X_estimated_EF_dist, P, d);

        % both E and F are corrupted and nyström on gram
        X_estimated_EF_gram = dist2gram(E_EF_Estimated, F_EF_Estimated, "gram");
        rmse_EF_gram(trial) = gram2rmse(X_estimated_EF_gram, P, d);

        % both E and F are corrupted and nyström on none
        X_estimated_EF_fullD = dist2gram(D_EF_estimated);
        rmse_EF_fullD(trial) = gram2rmse(X_estimated_EF_fullD, P, d);

        % only D is corrupted and nyström on none
        X_estimated_D = dist2gram(D_D_estimated);
        rmse_D(trial) = gram2rmse(X_estimated_D, P, d);
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
