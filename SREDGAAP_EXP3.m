function [Xk, Pk, timer] = SREDGAAP_EXP3(data, params)
    
   
    timer = 0;
    E = data.E_true;
    F_obs = data.F_obs;
    F_star = data.F_true;   
    P_star = data.P_true;

    [m, n] = size(F_obs);


    tol = get_field(params, 'tol', 1e-14);
    max_iter = get_field(params, 'max_iter', 100);
    show_output = get_field(params, 'show_output', 0);
    zeta0 = get_field(params, 'zeta0', 1.2 * max(F_star(:)));
    gamma = get_field(params, 'gamma', 0.9);
    d = get_field(params, 'd', 2); % point dimension
    r = get_field(params, 'r', d+2); % gram dimension
    accelerated = get_field(params, 'accelerated', true); % use accelerated projection

    known_row_id = get_field(params, 'known_row_id', 1); % known row id
    known_row_F = F_star(known_row_id, :); % known row of F_star
    F_obs(known_row_id, :) = known_row_F; % set the known row in F_obs
    
    % initialization 
    % Hard threshold to find initial outliers
    S0 = T_hardThreshold(F_obs, zeta0);

    % Compute initial L^1
    % Fk = F_obs - S0;
    Fk = F_star + randn(size(F_star)) * 10; % initial guess for Fk
    B0 = operatorB(Fk, E);  
    Bk = projHr(B0, d);
    Sk = S0;

    % display initial errors 
    if show_output == 2
        [~, Pk] = get_gram_and_points(E, Bk, d);
        err_P = Compute_RMSE(P_star', Pk);
        err_FoFs = norm(F_obs - F_star, 'fro') / norm(F_star, 'fro');
        err_FkFs = norm(Fk - F_star, 'fro') / norm(F_star, 'fro');
        fprintf('Initial: err_P = %.4g, err_FoFs = %.4g, err_FkFs = %.4g\n', err_P, err_FoFs, err_FkFs);            
    end

    for k=1:max_iter
        % Update threshold
        zeta_k = zeta0 * (gamma^(k-1));
        
        % Update outlier matrix
        ABk = operatorA(Bk, E, known_row_F, known_row_id);
        Sk_new = T_hardThreshold(F_obs - ABk, zeta_k);

        % compute new BK 
        Fk = F_obs - Sk_new; % new Fk
        Bk_new = operatorB(Fk, E);

        % project 
        if accelerated
            [Uk, ~, Vk] = svds(Bk_new, d); % SVD of Bk_new
            Bk_new = projTangent(Bk_new, Uk, Vk); % project onto tangent space
        end
        Bk_new = projHr(Bk_new, d); % project onto rank-d space

        diffBk = norm(Bk_new - Bk, 'fro') / norm(Bk, 'fro'); % relative difference
        
        Bk = Bk_new; % update Bk

        if k > 2000 && diffBk < tol                
            break;
        end

        if show_output == 2
            % compute errors
            [~, Pk] = get_gram_and_points(E, Bk, d);
            err_P = Compute_RMSE(P_star', Pk);
            err_FkFs = norm(Fk - F_star, 'fro') / norm(F_star, 'fro');
            fprintf('Iter %d: err_P = %.4g, err_FkFs = %.4g, diffBk = %.4g\n', k, err_P, err_FkFs, diffBk);       
        end
        
    end

    % final output
    [Xk, Pk] = get_gram_and_points(E, Bk, d); % get points from Gram matrix
end

%% =============== Helper Subfunctions ===============

function [Xk, Pk] = get_gram_and_points(E, Bk, d)
    A = compute_A(E);
    C = Bk' * pinv(A, 0.01) * Bk;
    Xk = [A Bk; Bk' C];
    Xk = projHrPlus(Xk, d); % project onto rank-d space     
    Pk = gram_to_points(Xk, d); % get points from Gram matrix  
end

function Zthr = T_hardThreshold(Z, zeta)
% T_hardThreshold   Hard thresholding operator
%   Zthr(i,j) = Z(i,j) if |Z(i,j)| > zeta, otherwise 0.
    Zthr = Z .* (abs(Z) > zeta);
end

function F = operatorA(B, E, known_row_F, k)
    % This function is used to map Gram matrix L -> EDM
    % operatorA(L) = diag(L)*1^T + 1*diag(L)^T - 2L
    % new: B to F 
    F = compute_F(B, E, known_row_F, k);
end


function B = operatorB(F, E)
    % This function is used to compute gram matrix from distance matrix
    % operatorB(Z) = -1/2 * J Z J
    % with J = I_n - (1/n) * 11^T
    % new: F to B

    B = compute_B(E, F);
end

function Zproj = projTangent(Z, Uk, Vk)
% projTangent:  P_{T^k}(Z) = Uk*Uk'^T Z + Z Uk*Uk'^T - Uk*Uk'^T Z Uk*Uk'^T
% Projects Z onto the tangent space of rank-r PSD manifold at L^k
    UkUt = Uk*(Uk');
    VkVt = Vk*(Vk');
    Zproj = UkUt*Z + Z*VkVt - UkUt*Z*VkVt;
end

function L = projHr(Z, r)
    % This function makes a rank-r projection of a matrix Z
    [U, S, V] = svds(Z, r);
    dvals = diag(S);
    dvals(dvals<0) = 0;  % Remove negative
    L = U*diag(dvals)*V';
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
