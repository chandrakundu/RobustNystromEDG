function [X_estimated, P_estimated, time_counter] = SREDG_AAP(data, params)

    % SREDG_AAP:  SREDG with Accelerated Alternating Projection (AAP) method

    E = data.E_true;
    F_corrupted = data.F_corrupted; 
    [m, n] = size(F_corrupted);
    F_star = data.F_true; 
    P_star = data.P_true;
    X_star = P_star'*P_star; 
    d = params.d; 
    r = d + 2; 



    % set default parameters
    if isfield(params, 'known_row_id')
        known_row_id = params.known_row_id; 
    else
        known_row_id = m; 
    end
    known_row_F = F_star(known_row_id, :); 
    F_corrupted(known_row_id, :) = known_row_F;

    if isfield(params, 'max_iter')
        max_iter = params.max_iter; 
    else
        max_iter = 100; 
    end

    if isfield(params, 'tol')
        tol = params.tol; 
    else
        tol = 1e-14; 
    end

    if isfield(params, 'show_output')
        show_output = params.show_output; 
    else
        show_output = 0; 
    end


    if isfield(params, 'zeta0')
        zeta0 = params.zeta0; % initial threshold for hard thresholding
    else
        zeta0 = 1 * max(F_corrupted(:)); % default to the maximum value of F_corrupted
    end

    if isfield(params, 'gamma')
        gamma = params.gamma; % decay rate for zeta
    else
        gamma = 0.8; % default to 0.9
    end

    if isfield(params, 'accelerated')
        accelerated = params.accelerated; % use accelerated method
    else
        accelerated = true; % default to false
    end

    if isfield(params, 'alpha')
        alpha = params.alpha; % percentage of outliers
    else
        alpha = 0.1; % default to 0.1
    end


    % initialize tracking variables
    timer = zeros(1, max_iter); % time tracking
    
    % test codes 
    


    % Initialization
    tic;

    % F_corrupted = GeometricConsistencyCleanup(E, F_corrupted); % geometric consistency cleanup
    zeta0 = max(abs(F_star(:))); 
    % S0 = hard_thresholding(F_star, zeta0);
    S0 = hard_thresholding(F_corrupted, zeta0); % hard thresholding   
    % sum(sum(S0 ~= 0))/(m*n)
    Fk = F_corrupted - S0; 

    B0 = operatorB(E, Fk); 
    Bk = projHr(B0, d);  

    init_timer = toc(tic);

    if show_output == 2
        [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);
        error_Fcor = norm(F_star - F_corrupted, 'fro') / norm(F_star, 'fro');
        error_Fk = norm(F_star - Fk, 'fro') / norm(F_star, 'fro');
        error_Bk = norm(B0 - Bk, 'fro') / norm(B0, 'fro');
        error_gram = norm(X_star - X_estimated, 'fro') / norm(X_star, 'fro');
        rmse_points = Compute_RMSE(P_star', P_estimated);
        fprintf('Initialization: error_Fobs_vs_Fstar = %0.4e, error_Fk_vs_Fstar = %0.4e, error_Bk = %0.4e, error_gram = %0.4e, rmse_points = %0.4e, time = %f seconds\n', error_Fcor, error_Fk, error_Bk, error_gram, rmse_points, init_timer);
    end



    for k = 1:max_iter
        tic;
        Fk = operatorA(Bk, E, known_row_F, known_row_id);
        
        Fk(known_row_id, : ) = known_row_F;
        % Fk_new = GeometricConsistencyCleanup(E, Fk_new);


        Rk = F_corrupted - Fk; 
        % zeta = zeta0 * (gamma^(k-1)); 
        % S0 = hard_thresholding(Rk, zeta); 
        

        % Zk = F_star - Fk_new;
        % better_zeta = max(abs(Zk(:)));

        % fprintf('zeta = %.4g, better_zeta = %.4g\n', zeta, better_zeta);
        
        % S0 = hard_thresholding(Rk, zeta); 
        % sum(sum(S0 ~= 0))/(m*n)




        Fk_new = F_corrupted - S0;
        % Fk_new = projHr(Fk_new, r); % projection onto rank-r manifold

        Bk_new = operatorB(E, Fk_new);

        % Project onto tangent space at Bk
        if accelerated
            Bk_new = projTangent(Bk_new, Bk, d); 
        end

        Bk_new = projHr(Bk_new, d);

        timer(k) = toc(tic); % time tracking
        if show_output == 2
            [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);            
            err_FvsF_star = norm(F_star - Fk, 'fro') / norm(F_star, 'fro');
            err_FvsFk = norm(Fk_new - Fk, 'fro') / norm(Fk, 'fro');
            error_Bk = norm(Bk - Bk_new, 'fro') / norm(Bk, 'fro');
            error_gram = norm(X_star - X_estimated, 'fro') / norm(X_star, 'fro');
            rmse_points = Compute_RMSE(P_star', P_estimated);
            fprintf('SREDG_AAP: i=%d, error_Fk_vs_Fstar = %.4f, error_Fk_vs_Fk_1 = %.4f, error_Bk = %.4f, error_gram = %.4f, rmse_points = %.4f, time = %f seconds\n', k, err_FvsF_star, err_FvsFk, error_Bk, error_gram, rmse_points, timer(k));
            
        end

        % if error_Bk < tol
        %     break;
        % end

        Bk = Bk_new; % update Bk
        Fk = Fk_new; % update Fk

    end
    time_counter = init_timer + sum(timer);  
    [X_estimated, P_estimated] = get_current_estimates(Bk, E, d);

    if show_output >= 1
        error_gram = norm(X_star - X_estimated, 'fro') / norm(X_star, 'fro');
        rmse_points = Compute_RMSE(P_star', P_estimated);
        fprintf('SREDG_AAP: i=%d, error_gram = %.4f, rmse_points = %.4f, time = %f seconds\n', k, error_gram, rmse_points, time_counter);
    end
end


function F = operatorA(B, E, known_row_F, k)
    F = compute_F(B, E, known_row_F, k);
    % F = computeF2(B,E);
end

function B = operatorB(E, F)
    B = compute_B(E, F);
end

function S = hard_thresholding(F, zeta)
    % HARD_THRESHOLDING Performs hard thresholding on the matrix F
    S = F .* (abs(F) >  zeta);
end

function Xr = projHr(X,r)
    % PROJHR Projects the matrix X onto the set of matrices with rank r
    % using Singular Value Decomposition (SVD)
    [U, S, V] = svd(X, 'econ');
    S = diag(S);
    S(r+1:end) = 0; % Set all singular values after r to zero
    Xr = U * diag(S) * V'; % Reconstruct the matrix with rank r
end

function Z_proj = projTangent(Z, Bk, d)
    % projTangent:  P_{T^k}(Z) = Uk*Uk'^T Z + Z Vk*Vk'^T - Uk*Uk'^T Z Vk*Vk'^T
    % Uk, Vk are the top-r left and right singular vectors of B^k
    % Projects Z onto the tangent space of rank-r manifold at B^k
        [Uk, ~, Vk] = svds(Bk, d);
        UkUt = Uk*(Uk');
        VkVt = Vk*(Vk');
        Z_proj = UkUt*Z + Z*VkVt - UkUt*Z*VkVt;
end
