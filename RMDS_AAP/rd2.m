function [Lk, Sk, Xk] = rd2(data, params)
    % RMDS_AAP  Robust MDS via Accelerated Alternating Projections.
    %
    %
    %   Inputs:
    %       D        - (n x n) observed EDM (symmetric)
    %       r        - target rank
    %       zeta0    - initial threshold parameter (scalar)
    %       gamma    - decay rate in (0,1)
    %       maxIter  - maximum number of iterations
    %       tol      - convergence tolerance (e.g., 1e-14)
    %
    %   Outputs:
    %       Lk       - final Gram matrix (n x n)
    %       Sk       - final outlier matrix (n x n)
    %       Xk       - points. factor of Lk, i.e. Lk ~= Xk * Xk', up to rounding
    %

        D = data.D_obs;
        L_star = data.X_true;  % true gram matrix (for error check only)
        X_star = data.P_true';  % true coordinates (for RMSE check only)

        tol = get_field(params, 'tol', 1e-14);
        maxIter = get_field(params, 'max_iter', 100);
        show_output = get_field(params, 'show_output', 0);
        zeta0 = get_field(params, 'zeta0', 1.2 * max(D(:)));
        gamma = get_field(params, 'gamma', 0.9);
        r = get_field(params, 'r', 2); % target rank


        % display error in observed gram matrix
        L_obs = operatorB(D);
        error_gram = norm(L_star - L_obs, 'fro') / max(1, norm(L_star,'fro'));
        error_points = Compute_RMSE(X_star, gram_to_points(L_obs, r));
        fprintf('Observed: Gram error: %e; \t Points RMSE: %e\n', error_gram, error_points);
        
        % Initialization
        % Hard threshold to find initial outliers
        S0 = T_hardThreshold(D, zeta0);
        
        % Compute initial L^1
        B0 = operatorB(D - S0);           % B = -1/2 * J(D-S0)J
        Lk = projHrPlus(B0, r);           % L^1 = H_r^+(B0)
        Sk = S0;                          % Current estimate of outlier matrix
        
        % gram matrix to points
        Xk = gram_to_points(Lk, r);

        % display initial error
        error_gram = norm(L_star - Lk, 'fro') / max(1, norm(L_star,'fro'));
        error_points = Compute_RMSE(X_star, Xk);
        fprintf('Initial Gram error: %e; Initial Points RMSE: %e\n', error_gram, error_points);
        
        % Main Loop
        for k = 1:maxIter
            
            % Update threshold
            zeta_k = zeta0 * (gamma^(k-1));
            
            % Update outlier matrix
            ALk = operatorA(Lk);
            Sk_new = T_hardThreshold(D - ALk, zeta_k);
            
            %  Compute new Lk
            Bk = operatorB(D - Sk_new);
            
            %   Project onto tangent space at L^k
            %   We need U^k from the rank-r decomposition of L^k to define P_{T^k}
            [Uk, ~] = eigs(Lk, r, 'largestreal', 'Tolerance',1e-6);            
            Bproj = projTangent(Bk, Uk);
            %   Then keep top-r PSD part
            Lk_new = projHrPlus(Bproj, r);
            
            % Convergence checks
            diffL = norm(Lk_new - Lk, 'fro') / max(1, norm(Lk,'fro'));
            Lk = Lk_new;
            Sk = Sk_new;
            
            %   update Xk 
            Xk = gram_to_points(Lk, r);

            % display error
            error_gram = norm(L_star - Lk, 'fro') / max(1, norm(L_star,'fro'));
            error_points = Compute_RMSE(X_star, Xk);
            fprintf('Iteration %d: \t Gram error: %e; \t Points RMSE: %e\n', k, error_gram, error_points);

            
            if diffL < tol
                fprintf('Converged at iteration %d with relative change %e\n', k, diffL);
                break;
            end
            
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
    