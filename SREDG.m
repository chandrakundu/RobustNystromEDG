function [X,P,time_counter] = SREDG(data, params)
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
    F = data.F_obs; % corrupted distance matrix
    F_true = data.F_true; % true distance matrix
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

    time_counter = 0;
    tstart = tic;
    [F_hat, ~] = RPCA(F, r, para );

    if nyston == "gram"    
        % compute A and B
        [A, B] = compute_AB(E, F_hat);

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
