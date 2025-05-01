clear;
load_directory;

% User-defined parameters
m_values = 30:10:40;
alpha_values = 0.1:0.1:0.2;
n_trials = 10; % Number of trials
res_file = "results/exp_sredgaap/temp_v01_aap_different_stopping_d2_p100_i2000.txt";
temp_file = "results/exp_sredgaap/temp_res_file2.txt";
desc = sprintf("Compare initial implementation AP with stoping criteria.\n");

p = 100;  % number of points
d = 2;    % dimension of the points

% Define methods to compare

% SREDG_AAP PARAMS
paramsAAP.show_output = 0; % Suppress output
paramsAAP.max_iter = 2000; % Maximum number of iterations
paramsAAP.tol = 1e-14; % Tolerance for convergence
paramsAAP.d = d; % Dimension of the points
paramsAAP.known_row_id = 1;
paramsAAP.accelerated = false;
paramsAAP.gamma = 0.95; 

paramsAP = paramsAAP;
% paramsAAP90.gamma = 0.9; % gamma = 0.9
% paramsAAP99 = paramsAAP;
% paramsAAP99.gamma = 0.99; % gamma = 0.92





methods = {
    struct('name', 'AAP_vanilla', ...
     'function', @SREDGAP, ....
     'params',  paramsAP, ...
    'desc', 'AAP_vanilla'),
    struct('name', 'AP_stopping', ...
     'function', @SREDGAP, ....
     'params',  paramsAP, ...
    'desc', 'SREDG AAP 99'),
};




% methods = {
%     struct('name', 'SREDGAAP', ...
%      'function', @SREDGAAP, ....
%      'params',  paramsAAP, ...
%     'desc', 'SREDG AAP new implementation from RMDSAAP'),
%     struct('name', 'SREDG_AAP', ...
%      'function', @SREDG_AAP, ....
%      'params',  paramsAAP, ...
%       'desc', 'SREDG AAP without projection'),
%     struct('name', 'SREDG', ...
%      'function', @SREDG, ....
%      'params',  params, ...
%       'desc', 'SREDG CISS paper')
% };

recovery_threshold = 1e-1;


results = struct();
trial_results = struct();
for method_idx = 1:length(methods)
    method = methods{method_idx};
    results.(method.name) = struct( ...
        'rmse', zeros(n_trials, length(m_values), length(alpha_values)), ...
        'mean_rmse', zeros(length(m_values), length(alpha_values)), ...
        'std_rmse', zeros(length(m_values), length(alpha_values)), ...
        'recovered', zeros(length(m_values), length(alpha_values)) ...
        );
end

fid_temp = fopen(temp_file, 'w');
fprintf(fid_temp, '### Temporary Comparison (num_trials = %d)\n', n_trials);

fprintf("Starting experiment...\n");

for i = 1:length(m_values)
    for j = 1:length(alpha_values)  
        fprintf("m = %d, alpha = %.2f ...", m_values(i), alpha_values(j));
        
        
        for trial = 1:n_trials
            % Generate data for this trial
            [F_obs, E_true, F_true, P_true] = generate_data_SREDG(alpha_values(j), m_values(i), p, d);

            data = struct( ...
                'E_true', E_true, ...
                'F_obs', F_obs, ...
                'P_true', P_true, ...
                'F_true', F_true ...
            );

            for method_idx = 1:length(methods)
                method = methods{method_idx};
                params = method.params;

                [~, P_estimated, ~] = method.function(data, params);
                [rmse, ~, ~] = Compute_RMSE(P_true', P_estimated);           
                results.(method.name).rmse(trial, i, j) = rmse;     
            end
        end
        
        fprintf(' (done)\n')

        fprintf(fid_temp, '\n#### m = %d, alpha = %.2f\n', m_values(i), alpha_values(j));
        fprintf(fid_temp, '| Method | Mean RMSE | Std RMSE | Recovered |\n');
        fprintf(fid_temp, '|--------|-----------|----------|-----------|\n');


        for method_idx = 1:length(methods)
            method = methods{method_idx};
            rmse_values = results.(method.name).rmse(:, i, j);
            rmse_sorted = sort(rmse_values);
            results.(method.name).mean_rmse(i, j) = mean(rmse_sorted(3:end-2));
            results.(method.name).std_rmse(i, j) = std(rmse_sorted);
            results.(method.name).recovered(i, j) = sum(rmse_values < recovery_threshold);

            fprintf(fid_temp, '| %s | %.4f | %.4f | %d |\n', ...
                method.name, ...
                results.(method.name).mean_rmse(i, j), ...
                results.(method.name).std_rmse(i, j), ...
                results.(method.name).recovered(i, j));

            fprintf("%s: Mean RMSE: %.4f, Std RMSE: %.4f, Recovered: %d\n", ...
                method.name, ...
                results.(method.name).mean_rmse(i, j), ...
                results.(method.name).std_rmse(i, j), ...
                results.(method.name).recovered(i, j));
        end
        
        fprintf("------------------------- \n");
    end
end


fclose(fid_temp);

% Write final results to markdown file
fid = fopen(res_file, 'w');
fprintf(fid, '%s\n', desc);
for method_idx = 1:length(methods)
    method = methods{method_idx};
    fprintf(fid, '**Method: %s(num_trials = %d)**\n', method.name, n_trials);
    fprintf(fid, '%s\n', method.desc);
    fprintf(fid, ' \n');
    write_markdown_table(fid, alpha_values, m_values, ...
        results.(method.name).mean_rmse, results.(method.name).std_rmse, results.(method.name).recovered);
    fprintf(fid, ' \n');
end
fclose(fid);
