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