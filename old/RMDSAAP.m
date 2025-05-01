function [X_estimated, P_estimated, time_counter] = RMDSAAP(data, params)

    D = data.D_obs; % observed data matrix
    X_star = data.X_true; % observed gram matrix 
    P_star = data.P_true; % true factor matrix
    d = data.d; % dimension of the points 
    r = d + 2; % number of factors
    time_counter = 0; 

    if isfield(params, "tol")
        tol = params.tol; 
    else
        tol = 1e-14; 
    end

    if isfield(params, "max_iter")
        max_iter = params.max_iter; 
    else
        max_iter = 100; 
    end

    if isfield(params, "show_output")
        show_output = params.show_output; 
    else
        show_output = 0; 
    end

    if isfield(params, "zeta0")
        zeta0 = params.zeta0; % initial threshold for hard thresholding
    else
        zeta0 = 1.2 * max(D(:)); % default to the maximum value of D
    end

    if isfield(params, "gamma")
        gamma = params.gamma; % decay rate for zeta
    else
        gamma = 0.9; % default to 0.9
    end

    % initialization 
    S0 = T_hardThreshold(D, zeta0); 

    X0 = operatorB(D - S0);  % X = -1/2 * J (D - S0) J
    Xk = projHrPlus(X0, r); 
    Sk = S0;

    Pk = gram_to_points(Xk, r); 

    % display initial error
    if show_output == 2
        err_gram = norm(Xk - X_star, 'fro') / max(1, norm(X_star, 'fro'));
        err_points = Compute_RMSE(P_star, Pk);
        fprintf('iter = 0, err_gram = %.4e, err_points = %.4e\n', err_gram, err_points);
    end

    % main loop 
    for k = 1:max_iter
        % update zeta
        zeta = zeta0 * gamma^(k-1); 

        % update S_k
        Dk = operatorA(Xk); % Dk = diag(Xk)*1^T + 1*diag(Xk)^T - 2*Xk
        Sk = T_hardThreshold(D - Dk, zeta); 

        % project onto the tangent space at Lk 
        [Uk, ~] = eigs(Xk, r, 'largestreal', 'Tolerance',1e-7); 
        Xk_new = projTangent(Lk, Uk);
        Xk_new = projHrPlus(Xk_new, r);


        % convergence check 
        diffXk = norm(Xk_new - Xk, 'fro') / max(1, norm(Xk, 'fro'));
        Xk = Xk_new;


        % display error
        if show_output == 2
            err_gram = norm(Xk - X_star, 'fro') / max(1, norm(X_star, 'fro'));
            err_points = Compute_RMSE(P_star, Pk);
            fprintf('iter = %d, err_gram = %.4e, err_points = %.4e, diffXk = %.4e\n', k, err_gram, err_points, diffXk);            
        end

        if diffXk < tol
            break; 
        end
    end

    % final estimates
    if show_output >= 1
        err_gram = norm(Xk - X_star, 'fro') / max(1, norm(X_star, 'fro'));
        err_points = Compute_RMSE(P_star, Pk);
        fprintf('Final estimates: err_gram = %.4e, err_points = %.4e\n', err_gram, err_points);
    end
end


    %% =============== Helper Subfunctions ===============
    
    function Zthr = T_hardThreshold(Z, zeta)
        % T_hardThreshold   Hard thresholding operator
        %   Zthr(i,j) = Z(i,j) if |Z(i,j)| > zeta, otherwise 0.
            Zthr = Z .* (abs(Z) > zeta);
        end
        
        function val = operatorA(L)
            % This function is used to map Gram matrix L -> EDM
            % operatorA(L) = diag(L)*1^T + 1*diag(L)^T - 2L
            d = diag(L);
            val = d*ones(1,length(d)) + ones(length(d),1)*d' - 2*L;
        end
        
        
        function Bval = operatorB(Z)
            % This function is used to compute gram matrix from distance matrix
            % operatorB(Z) = -1/2 * J Z J
            % with J = I_n - (1/n) * 11^T
        
            n = size(Z,1);
            J = eye(n) - (1/n) * ones(n);
            Bval = -0.5 * J * Z * J;
        end
        
        function Zproj = projTangent(Z, Uk)
        % projTangent:  P_{T^k}(Z) = Uk*Uk'^T Z + Z Uk*Uk'^T - Uk*Uk'^T Z Uk*Uk'^T
        % Projects Z onto the tangent space of rank-r PSD manifold at L^k
            UkUt = Uk*(Uk');
            Zproj = UkUt*Z + Z*UkUt - UkUt*Z*UkUt;
        end
        
        function Lplus = projHrPlus(Z, r)
        % This function makes a rank-r projection of a matrix Z onto the PSD cone. so Z has rank at most r and is PSD
        % projHrPlus :  H_r^+(Z) = Sum_{i=1 to r} max(lambda_i, 0) * u_i u_i^T
        %   1) Do a partial eigendecomposition to get the top r eigenpairs
        %   2) Truncate negative eigenvalues
            [Utmp, Dtmp] = eigs(Z, r, 'largestreal', 'Tolerance',1e-7);
            dvals = diag(Dtmp);
            dvals(dvals<0) = 0;  % Remove negative
            Lplus = Utmp*diag(dvals)*Utmp';
        end
        
        function Xmat = gram_to_points(L, r)
        % formXfromL: from a PSD matrix L (rank <= r), extract X s.t. L ~= X X'
        % For simplicity, do a partial eigen-decomposition:
            [Utmp, Dtmp] = eigs(L, r, 'largestreal', 'Tolerance',1e-7);
            dvals = max(diag(Dtmp), 0); % clip negative
            Xmat  = Utmp*diag(sqrt(dvals));
        end
        
