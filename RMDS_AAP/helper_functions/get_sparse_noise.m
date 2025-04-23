function F_corrupted = get_sparse_noise(F, alpha)
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
