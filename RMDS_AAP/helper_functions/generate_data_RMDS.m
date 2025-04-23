function [D_obs, X_true, P_true, D_true] = generate_data_RMDS(alpha, p, d, seed)
    % GENERATE_DATA Generates random sensor and anchor locations
    %
    % Inputs:
    % alpha: percentage of outliers
    % p: total number of points
    % d: dimension of the points
    % seed: random seed for reproducibility
    
    switch nargin
        case 0
            alpha = 0.1; % default value
            p = 101;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 1
            p = 101;     % default value
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 2
            d = 2;       % default value
            seed = 'shuffle'; % default value
        case 3
            seed = 'shuffle'; % default value
    end

    c = [6;6];
    rng(seed);  % Set the random seed for reproducibility
    [P, Pc] = generatePlusSign(p, c); % here Pc is centered at (0,0)
    P_true = Pc'; 

    X_true = Pc * Pc'; % Gram matrix

    D_true = diag(X_true) * ones(1,p) + ones(p,1) * diag(X_true)' - 2 * X_true; % squared distance matrix

    % outliers 
    D_obs = add_outliers(D_true, alpha); % Add noise to the distance matrix 
end



% figure;
% plot(X(:,1), X(:,2), 'b.');
% title('Centered Data');
% xlabel('x_1');
% ylabel('x_2');
% grid on;
% axis equal;


function [X, Xc] = generatePlusSign(n, c)
    % Generate 101 points in a "plus" shape, centered at c 
    % The plus shape consists of a horizontal line (with nHalf points) 
    % and a vertical line (with n - nHalf points) to avoid duplicating the center.

    
    
    nHalf = floor((n+1)/2); % if n = 101, nHalf = 51; if n = 100, nHalf = 50
    nVert = n - nHalf;       % will be 50
    
    coords = zeros(n, 2);
    
    % Horizontal line: from x = -19 to x = 31 at y = 6
    xHoriz = linspace(-19, 31, nHalf);
    yHoriz = 6 * ones(1, nHalf);
    
    % Vertical line: from y = -19 to y = 31 with nVert points at x = 6
    yVert = linspace(-19, 31, nVert);
    xVert = 6 * ones(1, nVert);
    
    % Assign coordinates
    coords(1:nHalf, 1) = xHoriz;
    coords(1:nHalf, 2) = yHoriz;
    coords(nHalf+1:end, 1) = xVert;
    coords(nHalf+1:end, 2) = yVert;
    
    X  = coords;                % raw points
    % Center the data at c
    Xc = bsxfun(@minus, X, c');  % Equivalent to X - ones(size(X,1),1)*c'
end


function D_obs = add_outliers(D, alpha)
    n = size(D,1);
    D_obs = D;
    N = (n*(n-1))/2;           % unique entries in the distance matrix
    numOut = round(alpha*N);       % # of outliers
    idx = randperm(N, numOut); % random set of corrupted pairs

    % We will treat the upper-triangular part (i<j). For convenience, store i<j indices
    [I, J] = find(triu(ones(n),1));  % all i<j
    for k = 1:numOut
       ik = I(idx(k));
       jk = J(idx(k));
       % Add large random outlier (e.g. scaled by ~10 times the largest distance)
       outlierVal = 10 * max(D(:)) * (1 + 0.5*rand());
       D_obs(ik, jk) = D_obs(ik, jk) + outlierVal;
       D_obs(jk, ik) = D_obs(ik, jk);  % keep symmetric
    end
end
