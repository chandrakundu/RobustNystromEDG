function F_clean = GeometricConsistencyCleanupShortestPath(E, F_hat)
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

        % using shortest pat 
        P2 = shortestPathAnchorTarget(dE, dF);
    
        for i = 1:m
            for j = 1:n
                outlier_votes   = 0;
    
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
                    end
                end
    
                % replace if enough votes and we have estimates
                if outlier_votes > vote_thresh && (fij < (1-delta)*lb || fij > (1+delta)*ub)
                    F_clean(i,j) = P2(i,j); 
                end
            end
        end
    end
    


    function P2 = shortestPathAnchorTarget(dE, dF)
        % dE : m×m linear anchor–anchor distances
        % dF : m×n linear anchor–target distances
        % P2 : m×n squared shortest-path distances anchor→target
        
        m = size(dE,1);
        n = size(dF,2);
        
        G = graph(dE);          % weighted, undirected
        
        % Pre-allocate
        P  = zeros(m,n);
        
        for j = 1:n
            % Add a temporary node 't' that connects to every anchor k with edge dF(k,j)
            Gtemp = addedge(G, m+1, 1:m, dF(:,j)');  % node m+1 is the target
            % Run Dijkstra once from all anchors to 't'
            dist_to_t = distances(Gtemp, 1:m, m+1);  % column vector length m
            P(:,j) = dist_to_t;                      % store linear distances
        end
        
        P2 = P.^2;
    end
        