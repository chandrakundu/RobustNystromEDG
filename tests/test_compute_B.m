
p = 500;  % number of points
d = 3;    % dimension of the points
m = 30;   % number of anchors, here minimum m = round(4*(d+2)*log(p));

n = p - m; % number of sensors

P1 = -100+200.*rand(d,m); 
P2 = -100+200.*rand(d,n); 

P = [P1 P2];  
% P = P - mean(P,2); % center the points

% Squared distance matrix
dist = squareform(pdist(P'));
D = dist.*dist;
r = d + 2; % rank of the distance matrix

% Gram matrix
X = compute_X_from_D(D, m);

% Blocks of D
E = D(1:m,1:m);
F = D(1:m,m+1:end);
G = D(m+1:end,m+1:end);

% Blocks of the Gram matrix
A = X(1:m,1:m);
B = X(1:m,m+1:end);
C = X(m+1:end,m+1:end);

B_estimated = compute_B(E, F);
B_estimated2 = compute_B2(E, F);
B_estimated3 = compute_B_elementwise(E, F);
errB2 = norm(B - B_estimated2, 'fro') / norm(B, 'fro');

errB = norm(B - B_estimated, 'fro') / norm(B, 'fro');
errB3 = norm(B - B_estimated3, 'fro') / norm(B, 'fro');


fprintf('Error in B: %.4f\n', errB);

fprintf('Error in B2: %.4f\n', errB2);
fprintf('Error in B3: %.4f\n', errB3);


% check if compute_A is correct
A_estimated = compute_A(E, F);
errA = norm(A - A_estimated, 'fro') / norm(A, 'fro');
fprintf('Error in A: %.4f\n', errA);




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

function B = compute_B2(E, F)
    % compute_B computes the matrix B using the given formula
    % Inputs:
    %   E - m x m matrix
    %   F - m x n matrix
    % Output:
    %   B - m x n matrix

    [mE, nE] = size(E);
    [mF, nF] = size(F);

    % Check matrix dimensions
    if mE ~= nE
        error('Matrix E must be square (m x m)');
    end
    if mE ~= mF
        error('Matrix E and F must have the same number of rows');
    end

    m = mE;
    n = nF;

    % Compute each term
    term1 = F;
    term2 = (1/m) * ones(m, m) * F;
    term3 = (1/m) * E * ones(m, n);
    mu_E = mean(E(:));
    term4 = mu_E * ones(m, n);

    % Final formula
    B = -0.5 * (term1 - term2 - term3 + term4);
end

function B = compute_B_elementwise(E, F)
    % compute_B_elementwise computes B using the element-wise formula
    % Inputs:
    %   E - m x m matrix
    %   F - m x n matrix
    % Output:
    %   B - m x n matrix

    [mE, nE] = size(E);
    [mF, nF] = size(F);

    if mE ~= nE
        error('E must be a square matrix (m x m)');
    end
    if mE ~= mF
        error('E and F must have the same number of rows');
    end

    m = mE;
    n = nF;

    % Precompute terms
    col_mean_F = mean(F, 1);     % 1 x n
    row_sum_E  = sum(E, 2);      % m x 1
    mu_E       = mean(E(:));     % scalar

    % Expand and broadcast terms
    term1 = F;                                % m x n
    term2 = repmat(col_mean_F, m, 1);         % m x n
    term3 = repmat(row_sum_E / m, 1, n);      % m x n
    term4 = mu_E * ones(m, n);                % m x n

    % Compute B
    B = -0.5 * (term1 - term2 - term3 + term4);
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


function X = compute_X_from_D(D, m)
    % compute_X computes X = -0.5 * (I - 1*s') * D * (I - s*1')
    % Inputs:
    %   D - n x n matrix
    %   m - number of initial entries in s that are set to 1/m
    % Output:
    %   X - n x n matrix

    [n1, n2] = size(D);
    if n1 ~= n2
        error('D must be a square matrix (n x n)');
    end
    n = n1;

    if m > n || m <= 0
        error('m must be a positive integer less than or equal to n');
    end

    % Construct s: first m entries are 1/m, rest are 0
    s = [ones(m,1) * (1/m); zeros(n - m, 1)];

    % Compute I, 1, and intermediate terms
    I = eye(n);
    one = ones(n, 1);

    % Apply the formula
    X = -0.5 * (I - one * s') * D * (I - s * one');
end
