clear;
load_directory;

% User-defined parameters
m_values = 30:10:30;
alpha_values = 0.1:0.1:0.3;
n_trials = 5; % Number of trials
res_file = "results/time_comp_v3_sredgapStopping_VS_sredg_d3_p500_i2000.txt";
temp_file = "results/temp_res_file2.txt";
desc = sprintf("p100 and d3 gamma .95 i2000 ");

p = 500;  % number of points
d = 3;    % dimension of the points

% Define methods to compare


% SREDG PARAMS
params.RPCA = @AccAltProj;
params.param_function = @get_rpca_params;
params.show_output = 0; % Suppress output
params.max_iter = 2000; % Maximum number of iterations
params.tol = 1e-14; % Tolerance for convergence
params.d = d; % Dimension of the points
params.nyston = "gram"; 

% SREDG_AAP PARAMS
paramsAAP.show_output = 0; % Suppress output
paramsAAP.max_iter = 2000; % Maximum number of iterations
paramsAAP.tol = 1e-14; % Tolerance for convergence
paramsAAP.d = d; % Dimension of the points
paramsAAP.known_row_id = 1;
paramsAAP.gamma = .95;


methods = {
    struct('name', 'SREDGAP', ...
     'function', @SREDGAP, ....
     'params',  paramsAAP, ...
    'desc', 'SREDG AP with stopping '),
    % struct('name', 'SREDG_AAP', ...
    %  'function', @SREDG_AAP, ....
    %  'params',  paramsAAP, ...
    %   'desc', 'SREDG AAP without projection'),
    struct('name', 'SREDG', ...
     'function', @SREDG, ....
     'params',  params, ...
      'desc', 'SREDG CISS paper')
};

recovery_threshold = 1;


results = struct();
trial_results = struct();
for method_idx = 1:length(methods)
    method = methods{method_idx};
    results.(method.name) = struct( ...
        'time', zeros(n_trials, length(m_values), length(alpha_values)), ...
        'rmse', zeros(n_trials, length(m_values), length(alpha_values)), ...
        'avg_time', zeros(length(m_values), length(alpha_values)), ...
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

                [~, P_estimated, totla_time] = method.function(data, params);
                [rmse, ~, ~] = Compute_RMSE(P_true', P_estimated);           
                results.(method.name).rmse(trial, i, j) = rmse; 
                results.(method.name).time(trial, i, j) = totla_time;    
            end
        end
        
        fprintf(' (done)\n')

        fprintf(fid_temp, '\n#### m = %d, alpha = %.2f\n', m_values(i), alpha_values(j));
        fprintf(fid_temp, '| Method | Mean RMSE | Std RMSE | Recovered | Avg. Time (s) |\n');
        fprintf(fid_temp, '|--------|-----------|----------|-----------|---------------|\n');


        for method_idx = 1:length(methods)
            method = methods{method_idx};
            rmse_values = results.(method.name).rmse(:, i, j);
            rmse_sorted = sort(rmse_values);
            results.(method.name).mean_rmse(i, j) = mean(rmse_sorted(3:end-2));
            results.(method.name).std_rmse(i, j) = std(rmse_sorted);
            results.(method.name).recovered(i, j) = sum(rmse_values < recovery_threshold);
            results.(method.name).avg_time(i, j) = mean(results.(method.name).time(:, i, j));

            fprintf(fid_temp, '| %s | %.4f | %.4f | %d | %.4f |\n', ...
                method.name, ...
                results.(method.name).mean_rmse(i, j), ...
                results.(method.name).std_rmse(i, j), ...
                results.(method.name).recovered(i, j), ...
                results.(method.name).avg_time(i, j));

            fprintf("%s: Mean RMSE: %.4f, Std RMSE: %.4f, Recovered: %d, Time: %.4f\n", ...
                method.name, ...
                results.(method.name).mean_rmse(i, j), ...
                results.(method.name).std_rmse(i, j), ...
                results.(method.name).recovered(i, j), ...
                results.(method.name).avg_time(i, j));
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
        results.(method.name).mean_rmse, results.(method.name).avg_time, results.(method.name).recovered);
    fprintf(fid, ' \n');
end
fclose(fid);
