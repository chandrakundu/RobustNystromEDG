function F_clean = GeometricConsistencyCleanup(E, F_hat)
    % Cleans noisy anchor–target squared distances via triangle‐inequality voting

        [m, n] = size(F_hat);
    
        % Parameters
        vote_thresh = ceil(m/3);
        eps_dist   = 1e-10;           % to avoid zero‐division
        F_clean    = F_hat;
        delta = .4; 
    
        % Precompute square‐roots
        dE = sqrt(E);
        dF = sqrt(max(F_hat, 0));     % clamp negatives to zero
    
        for i = 1:m
            for j = 1:n
                outlier_votes   = 0;
                valid_estimates = [];
    
                for k = 1:m
                    if k == i, continue; end
    
                    dik = dE(i,k);
                    dkj = dF(k,j);
                    if dik < eps_dist || dkj < eps_dist || dF(i,j) < eps_dist
                        continue;
                    end
    
                    % triangle‐inequality squared bounds
                    lb = max(0, (dik - dkj)^2);
                    ub =       (dik + dkj)^2;
    
                    fij = F_hat(i,j);
                    if fij < lb || fij > ub
                        outlier_votes = outlier_votes + 1;
    
                        % law‐of‐cosines 
                        cos_ikj = ( E(i,k) + fij - F_hat(k,j) ) ...
                                  / (2 * dik * sqrt(fij));
                        cos_ikj = min(1, max(-1, cos_ikj));

                        est = dik^2 + fij - 2*dik*sqrt(fij)*cos_ikj;
                        valid_estimates(end+1) = max(0, est);
                        % valid_estimates(end+1) = (lb + ub)/2;
                    end
                end
    
                % replace if enough votes and we have estimates
                if outlier_votes > vote_thresh && (fij < (1-delta)*lb || fij > (1+delta)*ub)
                    F_clean(i,j) = median(valid_estimates);
                end
            end
        end
    end
    