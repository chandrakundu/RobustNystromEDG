
m = 50;  n = 500;  r = 3;   alpha = .20;
k = ceil(alpha*m);           % = 10 corrupted rows

% --- synthetic data -----------------------------------------------------
rng(0);
U = randn(m,r);  V = randn(n,r);
B_star = U*V';

rows_active = randperm(m,k);
S_star = zeros(m,n);
S_star(rows_active,:) = 10*randn(k,n);

J = eye(m) - ones(m)/m;
S_prime = -0.5 * J * S_star;      % observed corruption
B1 = B_star + S_prime;

% --- AltProj ------------------------------------------------------------
[L_hat, S_hat] = altproj_rowhard(B1, r, k);

fprintf('\n=== Summary ===\n');
fprintf('Rel-err L : %.3e\n',  norm(L_hat-B_star,'fro')/norm(B_star,'fro'));
fprintf('Rel-err S: %.3e\n', norm(S_hat-S_prime,'fro')/norm(S_prime,'fro'));


function [L,S] = altproj_rowhard(B1, r, k, tol, maxIter)
    % Robust PCA with hard row-thresholding (AltProj style)
    %   B1      : m×n data matrix  (low-rank + few corrupted rows)
    %   r       : target rank of L
    %   k       : max corrupted rows  (use k = ceil(alpha*m))
    %   tol     : relative residual tolerance
    %   maxIter : outer-loop cap
    
        if nargin < 4, tol = 1e-7;    end
        if nargin < 5, maxIter = 500; end
    
        [m,~] = size(B1);
        normB = norm(B1,'fro');
    
        % --- initial low-rank via truncated SVD -----------------------------
        [U,Sig,V] = svds(B1, r);    L = U*Sig*V';
        S = zeros(size(B1));
    
        for t = 1:maxIter
            % --- sparse row support ----------------------------------------
            R = B1 - L;
            rowNorm = sqrt(sum(R.^2,2));        % ℓ2 per row
            [~,idx] = maxk(rowNorm, k);         % support rows
            S(:) = 0;                           % reuse memory
            S(idx,:) = R(idx,:);
    
            % --- low-rank projection ---------------------------------------
            [U,Sig,V] = svds(B1 - S, r);
            L = U*Sig*V';
    
            % --- stopping ---------------------------------------------------
            res = norm(B1 - L - S,'fro') / normB;
            fprintf('Iter %3d:  residual = %.3e\n', t, res);
            if res < tol, break; end
        end
    end
    