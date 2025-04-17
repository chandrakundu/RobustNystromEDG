function A = compute_A(E)
    % compute_A Computes block A of the Gram matrix from E block of distance matrices

    % Dimension of E 
    m = size(E, 1);

    % Vector and matrix of ones
    ones_m = ones(m, 1);
    ones_mm = (1/m) * (ones_m * ones_m');

    A = -0.5 * (E - E * ones_mm - ones_mm * E + m*mean(E(:)) * ones_mm);
end