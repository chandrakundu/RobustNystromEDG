function P = D2X(D, r)
       
    % Step 1: Get the number of points (assumes D is square)
    n = size(D, 1);
    
    % Step 2: Construct the centering matrix J
    J = eye(n) - (1/n) * ones(n, n);
    
    % Step 3: Compute the Gram matrix X
    X = -0.5 * J * D * J;
    
    % Step 4: Compute the r-truncated eigenvalue decomposition of X
    [U, Lambda] = eigs(X, r, 'lm');

    % Step 5: Compute the embedding matrix P
    P = U * sqrt(Lambda);
end
