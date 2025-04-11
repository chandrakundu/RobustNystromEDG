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

k = m; 
known_row_F = F(k, :); % known row of F

F_estimated = compute_F(B, E, known_row_F, k);

errF = norm(F - F_estimated, 'fro') / norm(F, 'fro');
fprintf('Error in F: %.4f\n', errF);


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
