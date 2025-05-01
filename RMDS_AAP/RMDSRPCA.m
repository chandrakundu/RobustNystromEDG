function [Lk, Xk, timer] = RMDSRPCA(data, params)
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
        timer = 0;
        D = data.D_obs;
        L_star = data.X_true;  % true gram matrix (for error check only)
        X_star = data.P_true';  % true coordinates (for RMSE check only)
        D_star = data.D_true;

        tol = get_field(params, 'tol', 1e-14);
        maxIter = get_field(params, 'max_iter', 100);
        show_output = get_field(params, 'show_output', 0);
        zeta0 = get_field(params, 'zeta0', 1.2 * max(D_star(:)));
        gamma = get_field(params, 'gamma', 0.9);
        r = get_field(params, 'd', 2); % target rank
        RPCA = get_field(params, 'RPCA', @AccAltProj);
        para = get_rpca_params(D_star, r+2);
        para.show_output = show_output;


        [D_hat,~] = RPCA(D, r+2, para);
        D_hat = D_hat - diag(diag(D_hat));
        
        Lk = operatorB(D_hat); 
        Lk = projHrPlus(Lk, r); 

        Xk = gram_to_points(Lk, r);



        error_gram = norm(L_star - Lk, 'fro') / max(1, norm(L_star,'fro'));
        error_points = Compute_RMSE(X_star, Xk);
        fprintf('Observed: Gram error: %e; \t Points RMSE: %e\n', error_gram, error_points);
        
        
        
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
    