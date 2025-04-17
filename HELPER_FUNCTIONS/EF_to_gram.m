function L = EF_to_gram(E, F)
    % EF_to_gram: Compute the Gram matrix from E and F blocks of distance matrices
    %   L = [A B; B' C]
    [A, B] = compute_AB(E, F);
    L = AB_to_gram(A, B);
end
