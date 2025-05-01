clear;
load_directory;

% User-defined parameters
p_values = 100:200:600;
alpha_values = 0.1:0.1:0.3;
d = 3;
n_trials = 50; % Number of trials
res_file = "results/RMDS_AAP_VS_RPCA_d3_data_RMDSSREDG_add_sparse.txt";
temp_file = "results/temp_res_file2.txt";


% Define methods to compare

% RMDSAAP PARAMS 
params.d = d; 


methods = {
    struct('name', 'RMDSAAP', ...
     'function', @RMDSAAP, ....
     'params',  params, ...
    'desc', 'RMDS AAP'), ...
    struct('name', 'RMDSRPCA', ...
     'function', @RMDSRPCA, ....
     'params',  params, ...
      'desc', 'RMDS RPCA')
};

recovery_threshold = 1e-1;


results = struct();
trial_results = struct();
for method_idx = 1:length(methods)
    method = methods{method_idx};
    results.(method.name) = struct( ...
        'rmse', zeros(n_trials, length(p_values), length(alpha_values)), ...
        'mean_rmse', zeros(length(p_values), length(alpha_values)), ...
        'std_rmse', zeros(length(p_values), length(alpha_values)), ...
        'recovered', zeros(length(p_values), length(alpha_values)) ...
        );
end

fid_temp = fopen(temp_file, 'w');
fprintf(fid_temp, '### Temporary Comparison (num_trials = %d)\n', n_trials);

fprintf("Starting experiment...\n");

for i = 1:length(p_values)
    for j = 1:length(alpha_values)  
        fprintf("m = %d, alpha = %.2f ...", p_values(i), alpha_values(j));
        
        
        for trial = 1:n_trials
            % Generate data for this trial
            [D_obs, X_true, P_true, D_true] = generate_data_RMDSSREDG(alpha_values(j), p_values(i), d);

            data = struct( ...
                'D_obs', D_obs, ... 
                'X_true', X_true, ...
                'P_true', P_true, ...
                'D_true', D_true ...
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

        fprintf(fid_temp, '\n#### m = %d, alpha = %.2f\n', p_values(i), alpha_values(j));
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
for method_idx = 1:length(methods)
    method = methods{method_idx};
    fprintf(fid, '**Method: %s(num_trials = %d)**\n', method.name, n_trials);
    fprintf(fid, '%s\n', method.desc);
    fprintf(fid, ' \n');
    write_markdown_table(fid, alpha_values, p_values, ...
        results.(method.name).mean_rmse, results.(method.name).std_rmse, results.(method.name).recovered);
    fprintf(fid, ' \n');
end
fclose(fid);
