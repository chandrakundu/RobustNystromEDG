function [F_obs, E_true, F_true, P_true] = generate_data_SREDG(alpha, m, p, d, seed)
    % GENERATE_DATA Generates random sensor and anchor locations
    %
    % Inputs:
    % alpha: percentage of outliers
    % p: total number of points
    % d: dimension of the points
    % seed: random seed for reproducibility
    
    switch nargin
        case 0
            alpha = 0.1; % default value
            p = 500;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 1
            p = 500;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 2
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 4
            seed = 'shuffle'; % default value
    end

    rng(seed);  % Set the random seed for reproducibility
    
    P = -100 + 200 * rand(d, p); 
    P_true = P - mean(P, 2);  % Center the points

    % Generate the Gram matrix
    X_true = P_true' * P_true; 
    
    % Compute squared distance matrix
    dist = squareform(pdist(P_true'));
    D_true = dist.^2;

    E_true = D_true(1:m, 1:m);
    F_true = D_true(1:m, m+1:end);

    
    F_obs = add_sparse_noise(F_true, alpha);  % Add noise to the distance matrix
    
end


function F_corrupted = add_sparse_noise(F, alpha)
    [m, n] = size(F);

    % Number of sparse noise entries
    num_noise = round(alpha * m * n);

    % Randomly select positions in F
    S_supp_idx = randperm(m * n, num_noise);

    % Noise range
    S_range = 10 * mean(mean(abs(F)));
    S_temp = 2 * S_range * rand(m, n) - S_range;

    % Sparse noise matrix
    S_true = zeros(m, n);
    S_true(S_supp_idx) = S_temp(S_supp_idx);

    F_corrupted = F + S_true;
end


function F_obs = add_outliers(F, alpha)
    [m, n] = size(F);
    F_obs = F;
    N = m * n;                        % total number of entries
    numOut = round(alpha * N);        % number of outliers
    idx = randperm(N, numOut);        % random linear indices

    % Add large random outlier (e.g. scaled by ~10 times the largest entry)
    outlierVals = 10 * max(F(:)) * (1 + 0.5*rand(numOut,1));
    F_obs(idx) = F_obs(idx) + outlierVals.'; % Transpose outlierVals to row
end
