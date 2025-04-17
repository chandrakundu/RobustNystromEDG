function P = gram_to_points(X, d)
    [V, Lam] = eigs(X, d, 'lm');
    P = V * sqrt(Lam);
end