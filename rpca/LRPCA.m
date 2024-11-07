function [X,times,errors,time_per_iter] = LRPCA(X_orig,Y,etas,zetas,params)  
    % input
    % U: m x r matrix
    % V: n x r matrix

    % comment 
    % eta(1) = 1/p, for other etas, we do not devide the step size or eta by p

    % output
    % L: m x r matrix 
    % R: n x r matrix
    % times: num_iter x 1 vector: cumulative times per iteration
    % errors: num_iter x 1 vector: error per iteration
    % time_per_iter: num_iter x 1 vector: time per iteration    
    

    % parameters
    if ~exist('params','var'), params=struct(); end
    if isfield(params,'r'), r=params.r; else r=5; end
    if isfield(params,'thresh_low'), thresh_low=params.thresh_low; else thresh_low=1e-6; end
    if isfield(params,'thresh_high'), thresh_high=params.thresh_high; else thresh_high=1e2; end
    if isfield(params,'error_change_thresh'), error_change_thresh=params.error_change_thresh; else error_change_thresh=1e-10; end
    if isfield(params, 'num_iter'), num_iter=params.num_iter; else num_iter = size(zetas, 2); end
    if isfield(params, 'verbose'), verbose=params.verbose; else verbose = 1; end   
        
    % verbose
    st = dbstack();
    if verbose == 1, fprintf("===============%s Started=============\n", st(1).name); end

    ss = norm(X_orig-Y,"fro"),norm(X_orig,"fro");
    disp(ss);



    % preparation
    time_counter = 0;  
    % time_per_iter = zeros(num_iter, 1);
    % times = zeros(num_iter, 1);
    % errors = zeros(num_iter, 1);   

    % initialization
    tstart = tic;
    [U0, Sigma0, V0] = lansvd(Y - Thre(Y, zetas(1)), r);
    L = U0*sqrt(Sigma0);
    R = V0*sqrt(Sigma0);
    tEnd = toc(tstart);
    time_counter = time_counter + tEnd;
    error = norm(L*R' - X_orig, 'fro')/norm(X_orig, 'fro');    
    times(1) = time_counter;   
    errors(1) = error;
    time_per_iter(1) = tEnd; 
    
    if verbose == 1, fprintf("k: 0 Err: %e Time: %f (Initialization) \n", error, time_counter);    end
    
    % loop
    for k=2:num_iter
        tstart = tic;
        X = L * R';
        S = Thre(Y - X, zetas(k));
        del_L = (X + S - Y) * R / (R' * R + eps('double') * eye(r));
        del_R = (X + S - Y)' * L / (L' * L + eps('double') * eye(r));
        L_plus = L - etas(k) * del_L;
        R_plus = R - etas(k) * del_R;          
        L = L_plus;
        R = R_plus;
        tEnd = toc(tstart);
        time_counter = time_counter + tEnd;
        X = L*R';
        error = norm(X - X_orig, 'fro')/norm(X_orig, 'fro');         
        errors(k) = error;
        times(k) = time_counter;
        time_per_iter(k) = tEnd;
        if verbose == 1, fprintf("k: %d Err: %e Time: %f\n", k-1, error, time_counter); end
        if error < thresh_low || error > thresh_high break; end
    end
    if verbose == 1, fprintf("======================================\n"); end
    if verbose == 2, fprintf("k: %d Err: %e Time: %f\n", k-1, error, time_counter); end 
end

% Thresholding Function
function S = Thre(S, theta)
    S = sign(S) .* max(abs(S) - theta, 0.0);
end
