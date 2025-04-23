function F = computeF2(B, E)
    % computeF  Recover squared‐distance block F from Gram blocks A,B via Nystrom
    %
    %   Input:
    %     A  – m×m anchor–anchor Gram block (must be invertible)
    %     B  – m×n anchor–target Gram block
    %   Output:
    %     F  – m×n squared‐distance block between anchors and targets
    %
    %   This implements
    %     C ≈ B' * (A \ B)            % Nystrom approximation of target–target Gram
    %     F_ij = A_ii + C_jj − 2*B_ij
    %
    
        % sizes
        [m, n] = size(B);
        A = compute_A(E); % m×m anchor–anchor Gram block
    
        % 1) Nystrom: approximate target–target Gram
        %    C ≈ Bᵀ A⁻¹ B
        C = B'*pinv(A, 0.01)*B;
    
        % 2) extract diagonals
        a = diag(A);    % m×1 vector: squared‐norms of anchors
        c = diag(C);    % n×1 vector: squared‐norms of targets
    
        % 3) build F via broadcasting
        %    F(i,j) = a(i) + c(j) − 2*B(i,j)
        F = repmat(a, 1, n) + repmat(c', m, 1) - 2*B;
    end
