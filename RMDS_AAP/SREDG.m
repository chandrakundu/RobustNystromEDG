function [X,P,timer] = SREDG(data, params)
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
    RPCA = get_field(params, 'RPCA', @AccAltProj);
    para = get_rpca_params(F_star, r);

    [F_hat, ~] = RPCA(F_obs, r, para);

    [A, B] = compute_AB(E, F_hat); 
    C = B' * pinv(A, 0.01) * B;
    X = [A B; B' C]; % initial Gram matrix
    X = projHrPlus(X, d); % project onto rank-d space
    P = gram_to_points(X, d); % get points from Gram matrix
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
    