function [X,P,time_counter] = SREDG_RPCAB(data, params)
  

    % extract data
    E = data.E_true;
    F = data.F_corrupted; % corrupted distance matrix
    F_true = data.F_true; % true distance matrix
    r = params.d + 2; % rank of the distance matrix
    d = params.d; % dimension of the points
    RPCA = params.RPCA; % Robust PCA algorithm
    paraF = params.param_function(F_true, r); % parameters for RPCA
    [m, n] = size(F); % number of anchors and sensors

    B_true = compute_B(E, F_true); % true block B
    paraB = params.param_function(B_true, d); % parameters for RPCA on B block


    if isfield(params, 'show_output')
        show_output = params.show_output;
        paraF.show_output = show_output;
        paraB.show_output = show_output;
    else
        show_output = 2;
    end

    if isfield(params, 'Bcorrection')
        Bcorrection = params.Bcorrection;
    else
        Bcorrection = true; % default to true
    end

    

    if isfield(params, 'nyston')
        nyston = params.nyston;
    else
        nyston = "gram";
    end

    time_counter = 0;
    tstart = tic;
    F_hat = GeometricConsistencyCleanup(E, F); 
    [F_hat, ~] = RPCA(F_hat, r, paraF );
    % F_hat = F;

    if nyston == "gram"    
        % compute A and B
        [A, B] = compute_AB(E, F_hat);


        [B_hat, SB] = RPCA(B, d, paraB); % RPCA on B block

        if Bcorrection == true
            Bcor = (1/m) * ones(m,1) * (ones(1,m) * SB);
        else
            Bcor = zeros(m, n);
        end
        B_hat = B_hat + Bcor; % add the mean back to B_hat
        % disp(max(SB(:)));
        

        % compute the Gram matrix
        C = B_hat'*pinv(A, 0.01)*B_hat;
        X = [A B_hat; B_hat' C];
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
