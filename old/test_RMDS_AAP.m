clear; clc;
load_directory
% Generate data
n = 101;
c = [6;6];
[X, Xc] = generatePlusSign(n, c);

% Compute Gram matrix and distance matrix
L_star = Xc*Xc';
D_star = diag(L_star)*ones(1,n) + ones(n,1)*diag(L_star)' - 2*L_star;

% inject outliers
alpha = 0.3;
D_obs = add_outliers(D_star, alpha);


% run RMDS_AAP
r = 2;
tol = 1e-14;
zeta0 = 1.1 * max(D_star(:));
gamma = 0.9;
maxIter = 100;
[Lk, Sk, Xk] = RMDS_AAP(D_obs, L_star, Xc, r, zeta0, gamma, maxIter, tol);


para = get_rpca_params(D_star, r);
[Lk2, ~] = AccAltProj(D_obs, r, para );

Xk2 = gram_to_points(Lk2, r);
error_gram = norm(L_star - Lk2, 'fro') / max(1, norm(L_star,'fro'));
error_points = Compute_RMSE(Xc, Xk2);
fprintf('RPCA RMDS \t Gram error: %e; \t Points RMSE: %e\n', error_gram, error_points);


% Visualize data
figure;
plot(X(:,1), X(:,2), 'b.');
hold on;
plot(c(1), c(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % mark the center
title('Original Data');
xlabel('x_1');
ylabel('x_2');
legend('Data Points', 'Center');
grid on;
axis equal;

figure;
plot(Xc(:,1), Xc(:,2), 'b.');
title('Centered Data');
xlabel('x_1');
ylabel('x_2');
grid on;
axis equal;

%%% ---------- HELPER: Construct "Plus Sign" data in 2D ----------
function [X, Xc] = generatePlusSign(n, c)
    % Generate 101 points in a "plus" shape, centered at c = [6;6]
    % The plus shape consists of a horizontal line (with nHalf points) 
    % and a vertical line (with n - nHalf points) to avoid duplicating the center.
    
    % n = 101 (assumed)
    nHalf = floor((n+1)/2); % will be 51
    nVert = n - nHalf;       % will be 50
    
    coords = zeros(n, 2);
    
    % Horizontal line: from x = -19 to x = 31 at y = 6
    xHoriz = linspace(-19, 31, nHalf);
    yHoriz = 6 * ones(1, nHalf);
    
    % Vertical line: from y = -19 to y = 31 with nVert points at x = 6
    yVert = linspace(-19, 31, nVert);
    xVert = 6 * ones(1, nVert);
    
    % Assign coordinates
    coords(1:nHalf, 1) = xHoriz;
    coords(1:nHalf, 2) = yHoriz;
    coords(nHalf+1:end, 1) = xVert;
    coords(nHalf+1:end, 2) = yVert;
    
    X  = coords;                % raw points
    % Center the data at c
    Xc = bsxfun(@minus, X, c');  % Equivalent to X - ones(size(X,1),1)*c'
end


function D_obs = add_outliers(D, alpha)
    n = size(D,1);
    D_obs = D;
    N = (n*(n-1))/2;           % unique entries in the distance matrix
    numOut = round(alpha*N);       % # of outliers
    idx = randperm(N, numOut); % random set of corrupted pairs

    % We will treat the upper-triangular part (i<j). For convenience, store i<j indices
    [I, J] = find(triu(ones(n),1));  % all i<j
    for k = 1:numOut
       ik = I(idx(k));
       jk = J(idx(k));
       % Add large random outlier (e.g. scaled by ~10 times the largest distance)
       outlierVal = 10 * max(D(:)) * (1 + 0.5*rand());
       D_obs(ik, jk) = D_obs(ik, jk) + outlierVal;
       D_obs(jk, ik) = D_obs(ik, jk);  % keep symmetric
    end
end
