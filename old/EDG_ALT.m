function [X_est, P_est, S_est] = EDG_ALT(E, F_corrupted, para)
% EDG_ALT   Alternating‐projections for anchor‐target localization
%
%   [X_est, P_est, S_est] = EDG_ALT(E, F_corrupted, para)
%
%   Inputs:
%     E             m×m  squared‐distance among anchors
%     F_corrupted   m×n  noisy squared‐distance between anchors & targets
%     para.r           scalar  embedding dimension (rank)
%     para.max_iter    (opt) # of AP iterations (default 50)
%     para.tau         (opt) soft‐threshold value (default = mean(|B_corr|))
%
%   Outputs:
%     X_est       (m+n)×(m+n) estimated Gram matrix
%     P_est       (m+n)×r   recovered coordinates in ℝ^r
%     S_est       m×n       estimated sparse‐noise matrix

    % 1. sizes and defaults
    [m, n] = size(F_corrupted);
    if ~isfield(para,'max_iter'), para.max_iter = 50; end
    r = para.r;
    
    % 2. build centering & compute A
    J = ones(m)/m;
    e_mean = mean(E(:));
    A = -0.5*(E - J*E - E*J + e_mean*ones(m));
    
    % 3. form centered cross‐Gram B_corr
    B_corr = -0.5*(F_corrupted ...
                   - J*F_corrupted ...
                   - (1/m)*E*ones(m,n) ...
                   + e_mean*ones(m,n));
    
    % 4. default tau if needed
    if ~isfield(para,'tau')
        para.tau = mean(abs(B_corr(:)));
    end
    tau = para.tau;
    max_iter = para.max_iter;
    
    % 5. initialize estimates
    B_est = B_corr;
    S_est = zeros(m,n);
    
    % 6. alternating projections loop
    for k = 1:max_iter
        % 6a. low‑rank projection via truncated SVD
        [U,Sig,V] = svd(B_corr - S_est, 'econ');
        L = U(:,1:r)*Sig(1:r,1:r)*V(:,1:r)';
        
        % 6b. sparse projection via soft‐thresholding
        R = B_corr - L;
        S_est = sign(R).*max(abs(R)-tau, 0);
        
        % 6c. update low‐rank part
        B_est = L;
    end
    
    % 7. estimate target–target block C_est
    C_est = B_est' * pinv(A) * B_est;
    
    % 8. assemble full Gram and enforce PSD rank‑r
    X_full = [ A,        B_est;
               B_est', C_est ];
    X_sym  = (X_full + X_full')/2;
    [Vall, Dall] = eig(X_sym);
    [dvals, idx] = sort(diag(Dall),'descend');
    dvals = max(dvals, 0);
    Ur = Vall(:, idx(1:r));
    Sr = diag(dvals(1:r));
    X_est = Ur * Sr * Ur';   % final Gram
    
    % 9. classical MDS to get coordinates
    P_est = Ur * sqrt(Sr);
end
