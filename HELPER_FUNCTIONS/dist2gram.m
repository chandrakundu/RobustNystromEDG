function X = dist2gram(D, m)
    p = size(D, 1);
    s = zeros(p,1);
    s(1:m) = 1/m;
    J = eye(p) - (ones(p,1)*s');
    X = -0.5*J*D*J';
end