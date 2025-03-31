% Example
clc; clear;

E = [1  2  3;
     4  5  6;
     7  8  9];  
F_original = [ 10  20  30   40;   
               12  22  32   42;
               15  25  35   45];

B = compute_B(E, F_original);

% Extract the k th row of F
k = 2;
kthRowF = F_original(k,:);  

% Reconstruct F using computeF
F_reconstructed = compute_F(B, E, kthRowF, k);

% Compare the reconstructed F with the original
disp('Original F:');
disp(F_original);

disp('Reconstructed F:');
disp(F_reconstructed);

% Check if they match 
difference_norm = norm(F_original - F_reconstructed, 'fro');
disp(['Difference (Frobenius norm): ', num2str(difference_norm)]);




function A = compute_A(E, F)
    % compute_A Computes block A of the Gram matrix from E block of distance matrices

    % Dimensions of E and F
    m = size(E, 1);

    % Vector and matrix of ones
    ones_m = ones(m, 1);
    ones_mm = (1/m) * (ones_m * ones_m');

    A = -0.5 * (E - E * ones_mm - ones_mm * E + m*mean(E(:)) * ones_mm);
end



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
