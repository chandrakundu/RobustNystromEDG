clear; 
load_directory;


% required inputs
m_values = 20:10:20;
alpha_values = 0.1:0.1:0.1;
n_trials = 10; % Number of trials
res_file = "sredg_aap_results/testing.txt"; 



% pretty much default parameters
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
write_markdown_table(fid, alpha_values, m_values, mn_new, std_new, recovered_new);

fprintf(fid, '#### Method: SREDG \n');
write_markdown_table(fid, alpha_values, m_values, mn_old, std_old, recovered_old);

% Close file
fclose(fid);

%% Trials
function [rmse_new, rmse_old] = run_trial_synthetic(m, alpha, n_trials)
    rmse_new = zeros(n_trials, 1);
    rmse_old = zeros(n_trials, 1);

    p = 500;  % number of points
    d = 2;    % dimension of the points
    r = d + 2; % rank of the distance matrix

    for trial = 1:n_trials
        [E, F_corrupted, P, E_true, F_true, G_true] = generate_data(alpha, m, p, d);
        
        
        [~, B] = compute_AB(E, F_corrupted); % compute A and B
   

        % apply SREDG
        para = get_rpca_params(F_true, d+2);
        para.show_output = 1;
        para.muB = 1.1*get_mu_kappa(B,r-2);

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
