function [E_out, F_out, P_out, E_true, F_true, G_true] = generate_data(alpha, m, p, d, seed)
    % GENERATE_DATA Generates random sensor and anchor locations
    %
    % Inputs:
    % alpha: percentage of outliers
    % m: number of anchors
    % p: total number of points
    % d: dimension of the points
    % seed: random seed for reproducibility
    
    switch nargin
        case 0
            alpha = 0.1; % default value
            m = 30;      % default value
            p = 500;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 1
            m = 30;      % default value
            p = 500;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 2
            p = 500;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 3
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 4
            seed = 'shuffle'; % default value
    end

    rng(seed);  % Set the random seed for reproducibility
    
    P1 = -100 + 200 * rand(d, m); 
    P2 = -100 + 200 * rand(d, p - m); 
    
    P = [P1, P2];  
    P = P - mean(P, 2);  % Center the points
    
    % Compute squared distance matrix
    dist = squareform(pdist(P'));
    D = dist.^2;
    
    E = D(1:m, 1:m);
    F = D(1:m, m+1:end);
    if nargout > 2
        G = D(m+1:end, m+1:end); 
    end
     
    switch nargout
        case 1
            E_out = E;
        case 2
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
        case 3
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
            P_out = P;
        case 4
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
            P_out = P;
            E_true = E;
        case 5
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
            P_out = P;
            E_true = E;
            F_true = F;
        case 6
            E_out = E;
            F_out = get_sparse_noise(F, alpha);
            P_out = P;
            E_true = E;
            F_true = F;
            G_true = G;
        otherwise
            error('Invalid number of output arguments. Expected 1 to 6 outputs.');
    end
end