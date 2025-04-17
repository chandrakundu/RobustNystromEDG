function [Lk, Xk] = get_current_estimates(Bk, E, r)
    Lk = EB_to_gram(E, Bk);
    Xk = gram_to_points(Lk, r);
    Xk_centered = Xk - mean(Xk, 1);
    Lk = Xk_centered * Xk_centered';
end