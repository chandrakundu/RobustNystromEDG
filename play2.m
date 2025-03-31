test_play2();

function test_play2()
    % Test which thresholding approach is faster: wthresh vs T_hardThreshold
    
    % Test parameters
    sizes = [10, 100, 1000, 2000, 5000]; % Matrix sizes to test
    num_runs = 10; % Number of runs for each size
    
    fprintf('Comparing performance of wthresh vs T_hardThreshold\n');
    fprintf('%-10s %-15s %-15s %-15s\n', 'Size', 'wthresh (ms)', 'Custom (ms)', 'Speedup');
    fprintf('%-10s %-15s %-15s %-15s\n', '----', '------------', '------------', '-------');
    
    for i = 1:length(sizes)
        n = sizes(i);
        
        % Generate random matrix
        Z = randn(n, n);
        zeta = 0.5;
        
        % Time wthresh
        t1 = 0;
        for j = 1:num_runs
            tic;
            Z1 = wthresh(Z, 'h', zeta);
            t1 = t1 + toc;
        end
        t1 = t1 * 1000 / num_runs; % Convert to milliseconds
        
        % Time custom function
        t2 = 0;
        for j = 1:num_runs
            tic;
            Z2 = T_hardThreshold(Z, zeta);
            t2 = t2 + toc;
        end
        t2 = t2 * 1000 / num_runs; % Convert to milliseconds
        
        % Verify results are identical
        if max(abs(Z1(:) - Z2(:))) > 1e-10
            error('Results from wthresh and T_hardThreshold are not identical');
        end
        
        % Calculate speedup
        speedup = t1 / t2;
        
        fprintf('%-10d %-15.3f %-15.3f %-15.3f\n', n, t1, t2, speedup);
    end
    
    % Memory usage comparison for large matrices
    if ~verLessThan('matlab','9.2') % For newer MATLAB versions with memory function
        try
            n = 10000;
            Z = randn(n, n);
            zeta = 0.5;
            
            fprintf('\nMemory usage comparison for %dx%d matrix:\n', n, n);
            
            m1 = memory;
            Z1 = wthresh(Z, 'h', zeta);
            m2 = memory;
            wthresh_mem = m2.MemUsedMATLAB - m1.MemUsedMATLAB;
            
            clear Z1;
            m1 = memory;
            Z2 = T_hardThreshold(Z, zeta);
            m2 = memory;
            custom_mem = m2.MemUsedMATLAB - m1.MemUsedMATLAB;
            
            fprintf('wthresh memory: %.2f MB\n', wthresh_mem/1024/1024);
            fprintf('Custom memory: %.2f MB\n', custom_mem/1024/1024);
            fprintf('Memory ratio: %.3f\n', wthresh_mem/custom_mem);
        catch
            fprintf('\nMemory usage comparison failed. This may not be supported in your MATLAB version.\n');
        end
    end
    
    fprintf('\nTest completed.\n');
end

function Zthr = T_hardThreshold(Z, zeta)
    % T_hardThreshold   Hard thresholding operator
    %   Zthr(i,j) = Z(i,j) if |Z(i,j)| > zeta, otherwise 0.
    Zthr = Z .* (abs(Z) > zeta);
end
