function [X,P,time_counter] = SREDG_EXP(data, params)
    % SREDG Sparse Robust Euclidean Distance Geometry algorithm for distance matrix completion
    %
    % Input:
    % E: m x m matrix, the distance matrix of anchor nodes
    % F: m x n matrix, the distance matrix between anchor and target nodes
    % r: the rank of the distance matrix (typically dimension + 2)
    % RPCA: the function handle of Robust PCA algorithm
    % para: the parameter for RPCA  (e.g., AccAltProj)
    %
    % Output:
    % X: (m+n) x (m+n) matrix, the Gram matrix of the distance matrix

    % clean the distance matrix between anchor and target nodes


    % extract data
    E = data.E_true;
    F_corrupted = data.F_corrupted; % corrupted distance matrix
    F_true = data.F_true; % true distance matrix
    d = params.d; % dimension of the points
    r = params.d + 2; % rank of the distance matrix

    if isfield(params, 'RPCA')
        RPCA = params.RPCA; % Robust PCA algorithm
    else
        RPCA = @AccAltProj; % default to AccAltProj
    end

    if isfield(params, 'param_function')
        para = params.param_function(F_true, r); % parameters for RPCA
    else
        para = get_rpca_params(F_true, r); % default to the parameters in params
    end




    if isfield(params, 'show_output')
        show_output = params.show_output;
        para.show_output = show_output;
    else
        show_output = 2;
    end

    if nargin < 6
        nyston = "gram";
    end

    if isfield(params, 'nyston')
        nyston = params.nyston; % method to compute the Gram matrix
    else
        nyston = "gram"; % default to "gram"
    end

    [m, n] = size(F_corrupted); % m: number of anchors, n: number of targets
    known_row_id = m; 
    known_row_F = F_true(known_row_id, :); % known row of F
    F_corrupted(known_row_id, :) = known_row_F; % replace the known row in the corrupted matrix
    B_true = compute_B(E, F_true); % true B matrix
    para = get_rpca_params(B_true, r); % parameters for RPCA

    time_counter = 0;
    tstart = tic;

    F_corrupted = GeometricConsistencyCleanup(E, F_corrupted); 

    diffB = compute_B_diff(E, F_corrupted, m); % compute B(i,j)-B(1,j) for all i,j

    B_hat = compute_B(E, F_corrupted); % compute B matrix from the corrupted F

    [diffB_hat, ~] = RPCA(diffB, d, para );
    [B_hat, ~] = RPCA(B_hat, d, para );

    B = zeros(m, n); % initialize B_hat
    for i = 1:m
        for j = 1:n
            B(i, j) = diffB_hat(i, j) + B_hat(known_row_id, j);
        end
    end





    if nyston == "gram"    
        % compute A and B
        % [A, B] = compute_AB(E, F_hat);
        A = compute_A(E); % compute A matrix from the anchor distances

        % compute the Gram matrix
        C = B'*pinv(A, 0.01)*B;
        X = [A B; B' C];
    else
        G = F_hat'*pinv(E, 0.01)*F_hat; 
        D_est = [E F_hat; F_hat' G]; % corrupted distance matrix
        X = dist2gram(D_est, size(E, 1)); % corrupted Gram matrix
    end

    tEnd = toc(tstart);
    time_counter = time_counter + tEnd;

    if show_output >= 1
        fprintf('SREDG: time = %f seconds\n', time_counter);
    end

    % fix the negative eigenvalues and ensure symmetry
    X = fix_gram_matrix(X);
    P = gram_to_points(X, r-2);
end


function D = compute_B_diff(E, F, k)
    % compute_B_diff  Compute B(i,j)-B(k,j) for all i,j using loops
    %   E : m×m matrix of anchor–anchor distances
    %   F : m×n matrix of anchor–target distances
    %   k : reference anchor index (integer in 1..m)
    %
    %   D : m×n matrix with D(i,j) = B(i,j) - B(k,j)

    [m, n] = size(F);

    % Precompute the row‐means of E
    rowMeanE = zeros(m,1);
    for i = 1:m
        rowMeanE(i) = sum(E(i, :)) / m;
    end

    % Allocate output
    D = zeros(m, n);

    % Compute each entry with nested loops
    for i = 1:m
        for j = 1:n
            term_i = F(i,j) - rowMeanE(i);
            term_k = F(k,j) - rowMeanE(k);
            D(i,j) = -0.5 * (term_i - term_k);
        end
    end
end
