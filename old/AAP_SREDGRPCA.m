function [Lk, Xk, time_counter] = AAP_SREDGRPCA(E, F, L_star, X_star, r, known_row_F, known_row_id, zeta0, gamma, max_iter, tol, show_output)
    % SREDG_AAP  Robust EDG  via Accelerated Alternating Projections.
    %
    %
    %   Inputs:
    %       E        - m x m matrix, the distance matrix of anchor nodes
    %       F        - m x n matrix, the distance matrix between anchor and target nodes
    %       L_star   - (m+n) x (m+n) matrix, the ground truth Gram matrix
    %       X_star   - (m+n) x d matrix, the ground truth points
    %       r        - target rank of the Gram matrix
    %       known_row_F: 1 x n vector, known row of F
    %       known_row_id: integer, index of the known row
    %       zeta0    - initial threshold parameter (scalar)
    %       gamma    - decay rate in (0,1)
    %       maxIter  - maximum number of iterations
    %       tol      - convergence tolerance (e.g., 1e-8)
    %
    %   Outputs:
    %       Lk       - final Gram matrix (n x n)
    %       Sk       - final outlier matrix (n x n)
    %       Xk       - points. factor of Lk, i.e. Lk ~= Xk * Xk', up to rounding    %

    if nargin < 12
        show_output = 0;
    end
    if nargin < 11
        tol = 1e-8;
    end

    err = -1*ones(max_iter,1);
    timer = -1*ones(max_iter,1);

    
    % 2) Initialization
    % Hard threshold to find initial outliers
    % S0 = T_hardThreshold(F, zeta0);

    tic;
    m = size(E, 1);
    n = size(F, 2);
    para.mu        = 1.1*get_mu_kappa(F,r);  
    para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
    para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
    para.trimming  = false;
    para.tol       = 1e-10;
    para.gamma     = 0.9;
    para.max_iter  = 500;
    para.show_output = 0;
    [Fk, ~] = AccAltProj(F, r+2, para );

    
    % Compute initial L^1
    % Fk = F - S0;
    B0 = operatorB(E,Fk);           
    Bk = projHr(B0, r);

    init_timer = toc;    
    
    if show_output == 2
        [Lk, Xk] = get_current_estimates(Bk, E, r);
        display_error(L_star, X_star, Lk, Xk, 0, init_timer, false);
    end

    factor_initial = 20;
    factor_final   = 5;
    % prev_median = NaN; 
    lambda = log(factor_initial/factor_final) / (max_iter - 1); % parameter for exponential decay
    
    % 3) Main Loop
    for k = 1:max_iter
        tic;
        % Update threshold
        % [Uk, Sigma, Vk] = svd(Bk,'econ');
        % zeta_k = zeta0 * (Sigma(r+1,r+1) + (gamma^k)*Sigma(1,1));
        % zeta_k = zeta0 * (gamma^(k));
        

        % applying operator A: B -> F
        Fk = operatorA(Bk, E, known_row_F, known_row_id);   
        Rk = F - Fk; 

        
        % zeta0 = zeta0 * 0.1;
        % zeta_k = zeta0 * (gamma^(k));

        % addaptive thresholding
        current_mad = mad(abs(Rk(:)),1);
        current_factor = factor_initial - (factor_initial - factor_final) * (k-1) / (max_iter-1); % linear decay 
        % current_factor = factor_initial * exp(-lambda*(k-1)); % exponential decay
        % if ~isnan(prev_median) && current_median > 1.1 * prev_median
        %     current_factor = 2*prev_factor; 
        % end
        zeta_k = current_factor * current_mad; 
        % prev_median = current_median;
        % prev_factor = current_factor;

        


        Sk_new = T_hardThreshold(Rk, zeta_k);
        
        % applying operator B: F -> B 
        Fk = F - Sk_new;
        Bk_new = operatorB(E, Fk);
        
        %   Project onto tangent space at Bk
        %   We need Uk, Vk from the rank-r decomposition of Bk to define P_{Tk}
        % [Uk, ~, Vk] = svds(Bk, r);       
        % Bk_new = projTangent(Bk_new, Uk, Vk);
        %   Then keep top-r 
        Bk_new = projHr(Bk_new, r);        


        timer(k) = toc;

        if show_output == 3
            err_Bk = norm(Bk - Bk_new, 'fro') / norm(Bk, 'fro');
            err_Fk = norm(Fk - F, 'fro') / norm(F, 'fro');
            [Lk, Xk] = get_current_estimates(Bk_new, E, r);
            error_gram = norm(L_star - Lk, 'fro') / norm(L_star,'fro');
            error_points = Compute_RMSE(X_star', Xk);
            fprintf('max_Rk: %.2f, med_Rk: %e \t', max(abs(Rk(:))),median(abs(Rk(:))));
            fprintf('i=%d; err_Bk: %e; err_Fk: %e; Gram error: %e; Points RMSE: %e; time: %f \n', k, err_Bk, err_Fk, error_gram, error_points, timer(k));
        end
        % Check convergence
        err(k) = norm(Bk - Bk_new, 'fro') / norm(Bk, 'fro');
        if err(k) < tol
            break;
        end

        

        % fprintf('Iteration %d; error: %e; time: %f \n', k, err(k), timer(k));

        % if err(k) < tol
        %     break;
        % end

        Bk = Bk_new;       
        
        % display error
        if show_output == 2
            [Lk, Xk] = get_current_estimates(Bk, E, r);
            display_error(L_star, X_star, Lk, Xk, k, timer(k), false);
        end        
    end
    time_counter = sum(timer(timer>0)) + init_timer;
    if show_output >= 1
        [Lk, Xk] = get_current_estimates(Bk, E, r);
        display_error(L_star, X_star, Lk, Xk, k, time_counter, true);
    end 
    if show_output == 0
        % Final output
        [Lk, Xk] = get_current_estimates(Bk, E, r);
    end 
end
    
    %% =============== Helper Subfunctions ===============

    function [Lk, Xk] = get_current_estimates(Bk, E, r)
        Lk = EB_to_gram(E, Bk);
        Xk = gram_to_points(Lk, r);
        Xk_centered = Xk - mean(Xk, 1);
        Lk = Xk_centered*Xk_centered';
    end

    function display_error(L_star, X_star, Lk, Xk, iteration, time_counter, final_error)
        error_gram = norm(L_star - Lk, 'fro') / norm(L_star,'fro');
        error_points = Compute_RMSE(X_star', Xk);
        if final_error
            fprintf('SREDG_AAP: Iteration %d; Gram error: %e; Points RMSE: %e;  Total Time: %f\n', iteration, error_gram, error_points, time_counter);
        else
            fprintf('Iteration %d; Gram error: %e; Points RMSE: %e; Time: %f\n', iteration, error_gram, error_points, time_counter);
        end
    end
    
    function Zthr = T_hardThreshold(Z, zeta)
    % T_hardThreshold   Hard thresholding operator
    %   Zthr(i,j) = Z(i,j) if |Z(i,j)| > zeta, otherwise 0.
        Zthr = Z .* (abs(Z) > zeta);
    end
    
    function F = operatorA(B, E, known_row_F, k)        
        F = compute_F(B, E, known_row_F, k);
    end
    
    
    function B = operatorB(E, F)
        B = compute_B(E, F);
    end
    
    function Zproj = projTangent(Z, Uk, Vk)
    % projTangent:  P_{T^k}(Z) = Uk*Uk'^T Z + Z Vk*Vk'^T - Uk*Uk'^T Z Vk*Vk'^T
    % Uk, Vk are the top-r left and right singular vectors of B^k
    % Projects Z onto the tangent space of rank-r manifold at B^k
        UkUt = Uk*(Uk');
        VkVt = Vk*(Vk');
        Zproj = UkUt*Z + Z*VkVt - UkUt*Z*VkVt;
    end
    
    % function Lplus = projHrPlus(Z, r)
    % % This function makes a rank-r projection of a matrix Z onto the PSD cone. so Z has rank at most r and is PSD
    % %   1) Do a partial eigendecomposition to get the top r eigenpairs
    % %   2) Truncate negative eigenvalues
    %     [Utmp, Dtmp] = eigs(Z, r, 'largestreal', 'Tolerance',1e-7);
    %     dvals = diag(Dtmp);
    %     dvals(dvals<0) = 0;  % Remove negative
    %     Lplus = Utmp*diag(dvals)*Utmp';
    % end

    function [F_r, sigma_1] = projHr(F, r)
        % projHr:  Project F onto rank-r matrix
        % [U, S, V] = svd(F, 'econ');
        % F_r = U(:,1:r) * S(1:r,1:r) * V(:,1:r)';
        [U, S, V] = svds(F, r);
        F_r = U * S * V';
        if nargout > 1
            sigma_1 = S(1,1);
        end
    end
    
    function Xmat = gram_to_points(L, r)
    % formXfromL: from a PSD matrix L (rank <= r), extract X s.t. L ~= X X'
    % For simplicity, do a partial eigen-decomposition:
        % [Utmp, Dtmp] = eigs(L, r, 'largestreal', 'Tolerance',1e-7);
        % dvals = max(diag(Dtmp), 0); % clip negative
        % Xmat  = Utmp*diag(sqrt(dvals));
        [V, Lam] = eigs(L, r, 'lm');
        Xmat = V*sqrt(Lam);
    end
    


    function A = compute_A(E, F)
        % compute_A Computes block A of the Gram matrix from E block of distance matrices
    
        % Dimensions of E and F
        m = size(E, 1);
    
        % Vector and matrix of ones
        ones_m = ones(m, 1);
        ones_mm = (1/m) * (ones_m * ones_m');
    
        A = -0.5 * (E - E * ones_mm - ones_mm * E + m*mean(E(:)) * ones_mm);
    end
    
    
    
    function B = compute_B(E, F)
        % COMPUTE_B Computes block B of the Gram matrix from E and F blocks of distance matrices
    
        % Dimensions of E and F
        m = size(E, 1);
        n = size(F, 2);
    
        % Vector of ones
        ones_m = ones(m, 1);
        ones_n = ones(n, 1);
    
        ones_mm = (1/m) * (ones_m * ones_m');
        ones_mn = (1/m) * (ones_m * ones_n');
    
        B = -0.5 * (F - ones_mm * F - E * ones_mn + m*mean(E(:)) * ones_mn);
    end
    
    function F = compute_F(B, E, known_row_F, k)
        % COMPUTE_F Computes the F block of the Gram matrix from B, E, and a known row of F
        %
        % Inputs:
        % B: m x n matrix, block B of the Gram matrix
        % E: m x m matrix, block E of the distance matrix
        % known_row_F: 1 x n vector, known row of F
        % k: integer, index of the known row
        %
        % Output:
        % F: m x n matrix, the F block of the Gram matrix
    
        % Dimensions of B
        [m, n] = size(B);
    
        rowSumE = sum(E, 2); 
        diffRowSumE = rowSumE - rowSumE(k);
    
        diffB = B - repmat(B(k,:), m, 1);
    
        FrowPart = repmat(known_row_F, m, 1);
    
        F = FrowPart - 2 * diffB + (1/m) * diffRowSumE * ones(1, n);
    end

    function L = EF_to_gram(E, F)
        % EF_to_gram:  Compute the Gram matrix from E and F blocks of distance matrices
        %   L = [A B; B' C]
        A = compute_A(E, F);
        B = compute_B(E, F);
        L = AB_to_gram(A, B);
    end

    function L = EB_to_gram(E, B)
        % EB_to_gram:  Compute the Gram matrix from E block of distance matrix and B block of gram matrix
        %   L = [A B; B' C]
        A = compute_A(E, B);
        L = AB_to_gram(A, B);
    end

    function L = AB_to_gram(A, B)
        % AB_to_gram:  Compute the Gram matrix from A and B blocks of gram matrix
        %   L = [A B; B' C]
        C = B'*pinv(A, 0.01)*B;
        L = [A B; B' C];
        L = fix_gram_matrix(L);
    end


    function X_new = fix_gram_matrix(X, tolerance)
        % FIX_GRAM_MATRIX Fixes the negative eigenvalues of the Gram matrix and ensures symmetry
        %
        % Input:
        % X: (m+n) x (m+n) matrix, the Gram matrix
        % tolerance: the tolerance for fixing the negative eigenvalues
        
        if nargin < 2
            tolerance = 0; 
        end
        
        % Eigen decomposition
        [V, D] = eig(X);
        
        % set those smaller than the tolerance to zero
        D = diag(D); 
        D(D < tolerance) = 0; 
        D = diag(D); 
        
        % Reconstruct
        X_new = V * D * V';
    
        % Ensure symmetry and real values
        % X_new = (X_new + X_new') / 2; 
        X_new = real(X_new);
    end
    