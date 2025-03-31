% Assuming E (m x m) and F (m x n) are already defined
clear;
function [A, B] = compute_gram_mat(E, F, tol)

    if nargin < 3
        tol = 0.01;
    end

    % Dimensions of E and F
    m = size(E, 1);
    n = size(F, 2);

    % Vector of ones
    ones_m = ones(m, 1);
    ones_n = ones(n, 1);

    % Row mean of E (column vector, m x 1)
    E_row_mean = mean(E, 2);

    % Column mean of E (row vector, 1 x m)
    E_col_mean = mean(E, 1);

    % Row mean of F (column vector, m x 1)
    F_col_mean = mean(F, 1);

    % Overall mean of E (scalar)
    E_mean = mean(E(:));

    % Compute A (m x m)
    A = -0.5 * (E - E_row_mean * ones_m' - ones_m * E_col_mean + E_mean * (ones_m * ones_m'));

    % Compute B (m x n)
    B = -0.5 * (F - ones_m * F_col_mean - E_row_mean * ones_n' + E_mean * (ones_m * ones_n'));

    X = [A B; B' B' * pinv(A, tol) * B];
end

function [A, B] = compute_gram_mat2(E, F, tol)

    if nargin < 3
        tol = 0.01;
    end

    % % Dimensions of E and F
    m = size(E, 1);
    n = size(F, 2);

    % Vector of ones
    ones_m = ones(m, 1);
    ones_n = ones(n, 1);

    ones_mm = (1/m) * (ones_m * ones_m');
    ones_mn = (1/m) * (ones_m * ones_n');

    A = -0.5 * (E - E * ones_mm - ones_mm * E + m*mean(E(:)) * ones_mm);
    B = -0.5 * (F - ones_mm * F - E * ones_mn + m*mean(E(:)) * ones_mn);
    % 
    % X = [A B; B' B' * pinv(A, tol) * B];




end



function [Gram_A, Gram_B] = compute_gram_ele(E,F, tol)
    if nargin < 3
        tol = 0.01;
    end
  
    m = size(E, 1);
    n = size(F, 2);
    % p = m+n;

    mean_E_all = mean(E, "all");    % Scalar (mean of all elements of E)
    mean_E_cols = mean(E, 1);       % Row vector, 1 x m (mean of each column of E)
    mean_E_rows = mean(E, 2);       % Column vector, m x 1 (mean of each row of E)
    mean_F_cols = mean(F, 1);       % Row vector, 1 x n (mean of each column of F)

    % calculate Gram matrix for anchor nodes
    mean_E_cols_exp = repmat(mean_E_cols, m, 1); % Repeat mean_E_cols for m rows (m x m)
    mean_E_rows_exp = repmat(mean_E_rows, 1, m); % Repeat mean_E_rows for m columns (m x m)

    Gram_A = -0.5 * (E - mean_E_cols_exp - mean_E_rows_exp + mean_E_all);

    % calculate Gram matrix for anchor-target nodes
    mean_E_rows_for_F = repmat(mean_E_rows, 1, n); % Expand mean_E_rows for F (m x n)
    mean_F_cols_exp = repmat(mean_F_cols, m, 1);   % Expand mean_F_cols for matrix subtraction in Gram_B (m x n)

    Gram_B = -0.5 * (F - mean_F_cols_exp - mean_E_rows_for_F + mean_E_all);

    X = [Gram_A Gram_B; Gram_B' Gram_B'*(pinv(Gram_A,tol))*Gram_B];

    % s = zeros(p,1);
    % s(1:m) = 1/m;
    % J = eye(p) - (ones(p,1)*s');
    % D = [E F;F' zeros(n,n)];
    % X = -0.5 * J*D*J';    
end


function X = compute_gram_direct(E,F, tol)
    if nargin < 3
        tol = 0.01;
    end
  
    m = size(E, 1);
    n = size(F, 2);
    p = m+n;

    s = zeros(p,1);
    s(1:m) = 1/m;
    J = eye(p) - (ones(p,1)*s');
    D = [E F;F' zeros(n,n)];
    X = -0.5 * J*D*J';
end

m = 10;
n = 90;
d = 2;
p = m + n;

P = rand(d, p);
X = P'*P;

D = squareform(pdist(P'));

E = D(1:m, 1:m);
F = D(1:m, m+1:end);

[A B] = compute_gram_mat(E, F);
[A2 B2] = compute_gram_ele(E, F);
[A3 B3] = compute_gram_mat2(E, F);
