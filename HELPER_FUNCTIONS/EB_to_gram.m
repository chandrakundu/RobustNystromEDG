function L = EB_to_gram(E, B)
    % EB_to_gram: Compute the Gram matrix from E block of distance matrix and B block of gram matrix
    %   L = [A B; B' C]
    A = compute_A(E);
    L = AB_to_gram(A, B);
end