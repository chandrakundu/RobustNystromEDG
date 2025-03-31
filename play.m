P = randn(10,2);
P_center = mean(P);
P = P - P_center;
L = P*P';
A = operatorA(L);
dist = squareform(pdist(P));
D = dist.*dist;
D_obs = add_outliers(D, 0.1);

L_hat = operatorB(D);

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
