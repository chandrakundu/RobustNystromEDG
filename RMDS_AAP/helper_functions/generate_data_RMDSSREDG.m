function [D_obs, X_true, P_true, D_true] = generate_data_RMDSSREDG(alpha, p, d, seed)
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
        case 3
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
    
    D_obs = add_sparse_noise(D_true, alpha);  % Add noise to the distance matrix

    
end


function F_corrupted = add_sparse_noise(F, alpha)
    m = size(F, 1);
    n = size(F, 2);

    % Only for square, symmetric, zero-diagonal matrices
    if m ~= n
        error('F must be square for symmetric noise addition.');
    end

    % Indices of upper triangle, excluding diagonal
    upper_idx = find(triu(ones(m), 1));
    num_upper = numel(upper_idx);

    % Number of sparse noise entries
    num_noise = round(alpha * num_upper);

    % Randomly select positions in the upper triangle
    S_supp_idx = randsample(upper_idx, num_noise, false);

    % Noise range
    S_range = 10 * mean(mean(abs(F)));
    S_temp = 2 * S_range * rand(m, n) - S_range;

    % Sparse noise matrix
    S_true = zeros(m, n);
    S_true(S_supp_idx) = S_temp(S_supp_idx);

    % Copy upper triangle to lower triangle for symmetry
    S_true = S_true + S_true';

    % Ensure diagonal remains zero
    S_true(1:m+1:end) = 0;

    F_corrupted = F + S_true;
end


function D_obs = add_outliers(D, alpha)
    n = size(D,1);
    D_obs = D;
    N = (n*(n-1))/2;           % unique entries in the distance matrix
    numOut = round(alpha*N);       % # of outliers
    idx = randperm(N, numOut); % random set of corrupted pairs

    % We will treat the upper-triangular part (i<j). For convenience, store i<j indices
    [I, J] = find(triu(ones(n),1));  % all i<j
    for k = 1:numOut
       ik = I(idx(k));
       jk = J(idx(k));
       % Add large random outlier (e.g. scaled by ~10 times the largest distance)
       outlierVal = 10 * max(D(:)) * (1 + 0.5*rand());
       D_obs(ik, jk) = D_obs(ik, jk) + outlierVal;
       D_obs(jk, ik) = D_obs(ik, jk);  % keep symmetric
    end
end
