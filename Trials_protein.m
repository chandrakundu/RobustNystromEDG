addpath rpca\

% "2lum", "5wov"
% protein_names = ["1ax8", "1ptq"];
% protein_names = ["2lum", "5wov"];
% protein_names = ["1w2e","1ubq_modified"];
protein_names = ["1w2e","1ptq"];

for pn = 1:length(protein_names)
    protein_name= protein_names(pn);
    protein_file = strcat("data/proteins/",protein_name,".pdb");
    res_file = strcat("results/res_",protein_name,"_tr50_v2.txt"); 
    outs_cell = pdb2mat(protein_file);


    PP = [outs_cell.X ;outs_cell.Y; outs_cell.Z];

    % Define parameters
    m_values = 10:10:60;
    alpha_values = 0.05:0.05:0.3;
    n_trials = 50; % Number of trials



    % Initialize results matrices
    rmse_matrix = zeros(length(m_values), length(alpha_values));
    std_matrix = zeros(length(m_values), length(alpha_values));

    % Run trials and store RMSE values
    for i = 1:length(m_values)
        for j = 1:length(alpha_values)
            [rmse, std_dev] = run_trial_protein(PP, m_values(i), alpha_values(j), n_trials, protein_name);
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
        fprintf(fid, '\n| %d (%d) |', m_values(i), length(PP));
        for j = 1:length(alpha_values)
            fprintf(fid, ' %.2f (%.2f) |', rmse_matrix(i, j), std_matrix(i, j));
        end
    end

    % Close file
    fclose(fid);
end

%% Trials
function [rmse, std_dev] = run_trial_protein(PP, m, alpha, n_trials,protein_name)

    rmses = zeros(n_trials, 1);

    for trial = 1:n_trials
        sz_P = size(PP);
        d= sz_P(1);
        p = sz_P(2);
        n = p - m;
        r = d+2;  % rank of the distance matrix

        anchor_indices = randperm(p, m);
        target_indices = setdiff(1:p, anchor_indices);

        P1 = PP(:, anchor_indices); % Anchors
        P2 = PP(:, target_indices); % Target nodes

        % Combine P1 and P2 to form new P
        P = [P1, P2];
        % P = PP; % Uncomment this line to not randomly select anchor points
        
        % Ground distance matrix
        dist = squareform(pdist(P'));
        D = dist.*dist;

        % Blocks of D
        % m = round(4*(d+2)*log(p));
        
        E = D(1:m,1:m);
        F = D(1:m,m+1:end);

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
        X_estimated = dist2gram(E, F_estimated);

        [V, Lam] = eigs(X_estimated, d, 'lm');
        P_estimated = V*sqrt(Lam);

        % rmse
        [rmses(trial), ~, ~] = Compute_RMSE(P',P_estimated);
        
        % if trial == n_trials
        %     idx_anchors = [anchor_indices target_indices];
        %     [~,idx_aligned] = ismember(1:p,idx_anchors);
        %     % Save to pdb
        %     outs_cell.X = P_estimated(idx_aligned,1);
        %     outs_cell.Y = P_estimated(idx_aligned,2);
        %     outs_cell.Z = P_estimated(idx_aligned,3);
        %     outs_cell.outfile = strcat('data/proteins/estimated/', protein_name, '_m', num2str(m), '_a', num2str(alpha), '.pdb');
        %     mat2pdb(outs_cell);
        % end
    end

    rmse = mean(rmses);
    std_dev = std(rmses);
end
