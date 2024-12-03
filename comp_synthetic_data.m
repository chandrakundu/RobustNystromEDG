clear;
addpath rpca\

% Define parameters
m_values = 10:10:60;
alpha_values = 0.1:0.1:0.3;
n_trials = 100; % Number of trials
res_file = "draft_res/res_synthetic_tr100_comp.txt";

% Initialize results matrices
rmse_matrix1 = zeros(length(m_values), length(alpha_values));
std_matrix1 = zeros(length(m_values), length(alpha_values));
rmse_matrix2 = zeros(length(m_values), length(alpha_values));
std_matrix2 = zeros(length(m_values), length(alpha_values));

% Run trials and store RMSE and standard deviation values
for i = 1:length(m_values)
    for j = 1:length(alpha_values)
        [rmse1, std_dev1, rmse2, std_dev2] = run_trial_synthetic(m_values(i), alpha_values(j), n_trials);
        rmse_matrix1(i, j) = rmse1; % Store RMSE in matrix
        std_matrix1(i, j) = std_dev1; % Store standard deviation in matrix
        rmse_matrix2(i, j) = rmse2; % Store RMSE in matrix
        std_matrix2(i, j) = std_dev2; % Store standard deviation in matrix
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
        % fprintf(fid, ' %.2e (%.2e) |', rmse_matrix(i, j), std_matrix(i, j));
        fprintf(fid, ' %.4f (%.4f) |', rmse_matrix1(i, j), std_matrix1(i, j));
    end
end

fprintf(fid, '\n');

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
        % fprintf(fid, ' %.2e (%.2e) |', rmse_matrix(i, j), std_matrix(i, j));
        fprintf(fid, ' %.4f (%.4f) |', rmse_matrix2(i, j), std_matrix2(i, j));
    end
end

% Close file
fclose(fid);

%% Trials
function [rmse1, std_dev1, rmse2, std_dev2] = run_trial_synthetic(m, alpha, n_trials)

    rmses1 = zeros(n_trials, 1);
    rmses2 = zeros(n_trials, 1);

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
        X_estimated1 = dist2gram_matrix(E, F_estimated, 0.01);
        X_estimated2 = dist2gram_matrix2(E, F_estimated, 0.01);

        [V1 Lam1] = eigs(X_estimated1, d, 'lm');
        P_estimated1 = V1*sqrt(Lam1);
        % rmse
        [rmses1(trial), ~, ~] = Compute_RMSE(P',P_estimated1);   
        
        [V2 Lam2] = eigs(X_estimated2, d, 'lm');
        P_estimated2 = V2*sqrt(Lam2);
        % rmse
        [rmses2(trial), ~, ~] = Compute_RMSE(P',P_estimated2);

    end

    rmse1 = mean(rmses1);
    std_dev1 = std(rmses1);

    rmse2 = mean(rmses2);
    std_dev2 = std(rmses2);

end
