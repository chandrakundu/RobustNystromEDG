clear;
addpath rpca\
addpath LIB\

% Define parameters
% m_values = 10:5:50;
% alpha_values = 0.1:0.05:0.3;
m_values = 20:10:40;
alpha_values = 0.1:0.1:0.3;
% m_values = 30:5:30;
% alpha_values = 0.2:0.05:0.2;
n_trials = 50; % Number of trials
res_file = "sredg_aap_results/comp_v1_3_sredg_rpcaB_vs_sredg_1.txt";
recovery_threshold = 1e-1; % Recovery threshold for RMSE

% Initialize results matrices
% here mn_new and mn_old are the mean RMSE values of the new and old methods
mn_new = zeros(length(m_values), length(alpha_values));
mn_old = zeros(length(m_values), length(alpha_values));

std_new = zeros(length(m_values), length(alpha_values));
std_old = zeros(length(m_values), length(alpha_values));

recovered_new = zeros(length(m_values), length(alpha_values));
recovered_old = zeros(length(m_values), length(alpha_values));


% Run trials and store RMSE and standard deviation values
for i = 1:length(m_values)
    for j = 1:length(alpha_values)
        [rmse_new, rmse_old] = run_trial_synthetic(m_values(i), alpha_values(j), n_trials);
        
        rmse_new_sorted = sort(rmse_new);
        rmse_old_sorted = sort(rmse_old);
        mn_new(i, j) = mean(rmse_new_sorted(3:end-2));
        mn_old(i, j) = mean(rmse_old_sorted(3:end-2));

        std_new(i, j) = std(rmse_new_sorted);
        std_old(i, j) = std(rmse_old_sorted);

        recovered_new(i, j) = sum(rmse_new < recovery_threshold);
        recovered_old(i, j) = sum(rmse_old < recovery_threshold);
    end
end

% markdown table generation
fid = fopen(res_file, 'w');

fprintf(fid, '### Comparison (num_trials = %d)\n', n_trials);
fprintf(fid, '#### Method: SREDG_RPCAB  \n');
fprintf(fid, ' \n');
% fprintf(fid, 'factor_initial = 20; factor_final = 5;   \n');
% fprintf(fid, 'MAD and linear decay  \n  zeta_k = factor * mad(abs(Rk(:)));  \n');
% fprintf(fid, '```\n no projection on Bk \n \n');
write_markdown_table(fid, alpha_values, m_values, mn_new, std_new, recovered_new);

fprintf(fid, '#### Method: SREDG \n');
write_markdown_table(fid, alpha_values, m_values, mn_old, std_old, recovered_old);

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

        P1 = -100+200.*rand(d,m); 
        P2 = -100+200.*rand(d,n); 

        P = [P1 P2]; 
        P = P - mean(P,2); 

        % gram matrix
        X = P'*P;
               
        % distance matrix
        dist = squareform(pdist(P'));
        D = dist.*dist;
        r = d + 2; % rank of the distance matrix

        % Blocks of D
        E = D(1:m,1:m);
        F = D(1:m,m+1:end);
        % G = D(m+1:end,m+1:end);
        F_corrupted = get_sparse_noise(F, alpha);
        [~, B] = compute_AB(E, F_corrupted); % compute A and B
   

        % % apply AAP_SREDG
        % known_row_id = m;
        % known_row_F = F(known_row_id,:);  
        % tol = 1e-14; 
        % zeta0 = 1 * max(F(:));
        % gamma = 0.9;
        % max_iter = 500;
        % accelerated = false;
        % show_output = 1;
        % [~, P_estimated_1, ~] = SREDG_AAP_EXP(E, F_corrupted, F, X, P, d, known_row_F, known_row_id, zeta0, gamma, accelerated, max_iter, tol,show_output);

        % accelerated = true;
        % [~, P_estimated_2, ~] = SREDG_AAP(E, F_corrupted, F, X, P, d, known_row_F, known_row_id, zeta0, gamma, accelerated, max_iter, tol,show_output);


        % apply SREDG
        % RPCA params
        para.mu        = 1.1*get_mu_kappa(F,r);  
        para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
        para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
        para.trimming  = false;
        para.tol       = 1e-14;
        para.gamma     = 0.9;
        para.max_iter  = 500;
        para.show_output = 1;
        para.muB       = 1.1*get_mu_kappa(B,r-2);

        [~, P_estimated_1, ~ ] = SREDG_RPCAB(E,F_corrupted,r,@AccAltProj, para, "gram");

        [~, P_estimated_2, ~ ] = SREDG(E,F_corrupted,r,@AccAltProj, para, "gram");


        % calculate RMSE 
        [rmse_1, ~, ~] = Compute_RMSE(P',P_estimated_1); 
        [rmse_2, ~, ~] = Compute_RMSE(P',P_estimated_2);

        rmse_new(trial) = rmse_1;
        rmse_old(trial) = rmse_2;
        fprintf('m = %d, alpha = %f, trial = %d, RMSE_new = %f, RMSE_old = %f\n================================\n', m, alpha, trial, rmse_1, rmse_2);
    end
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


function [A, B] = compute_AB(E, F)
    % COMPUTE_AB Computes blocks A and B of the Gram matrix from E and F blocks of distance matrices

    % Dimensions of E and F
    m = size(E, 1);
    n = size(F, 2);

    % Vector of ones
    ones_m = ones(m, 1);
    ones_n = ones(n, 1);

    ones_mm = (1/m) * (ones_m * ones_m');
    ones_mn = (1/m) * (ones_m * ones_n');

    A = -0.5 * (E - E * ones_mm - ones_mm * E + m*mean(E(:)) * ones_mm);
    B = -0.5 * (F - ones_mm * F - E * ones_mn + m*mean(E(:)) * ones_mn);
end
