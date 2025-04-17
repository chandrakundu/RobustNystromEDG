function L = AB_to_gram(A, B)
    % AB_to_gram: Compute the Gram matrix from A and B blocks of gram matrix
    %   L = [A B; B' C]
    C = B' * pinv(A, 0.01) * B;
    L = [A B; B' C];
    L = fix_gram_matrix(L);
end