function [ L, S ] = AccAltProj( D, r, para )
% [ L, S ] = AccAltProj( D, r, para )
% 
% Inputs:
% D : Observed matrix. Sum of underlying low rank matrix and underlying
%     sparse matrix. 
% r : Target rank of underlying low rank matrix.
% params : parameters for the algorithm
%   .max_iter : Maximum number of iterations. (default 50)
%   .tol : Desired Frobenius norm error. (default 1e-5)
%   .beta_init : Parameter for thresholding at initialization. (default
%                4*beta)
%   .beta : Parameter for thresholding. (default 1/(2*nthroot(m*n,4)))
%   .gamma : Parameter for desired convergence rate. Value should between 0
%            and 1. Turn this parameter bigger will slow the convergence
%            speed but tolerate harder problem, such as higher p, r or mu. 
%            (default 0.5)   
%   .trimming : Determine whether using trimming step. (default false)
%   .mu : Incoherence of underlying low rank matrix. Input can be in format
%         of .mu = mu_max, or .mu = [mu_U, mu_V]. (default 5) 
%
% Outputs:
% L : Estimated low rank component of D.
% S : Estimated sparse component of D.
%
%
% Please cite the paper "Accelerated Alternating Projections for Robust
% Principal Component Analysis" if you find this code helpful
%
%
% Warning: We found this code runs very slow on AMD CPUs with earlier 
% versions of Matlab. For best experience, please use the code on Intel CPU
% based computer, and update Matlab to the latest version.
%
%
% By
% HanQin Cai         , Jian-Feng Cai, Ke Wei
% hqcai@math.ucla.edu, jfcai@ust.hk , kewei@fudan.edu.cn

% if exist('./rpca/PROPACK', 'dir')==7
%     addpath E:\projects\edg\RobustNystromEDG\rpca\PROPACK\;
%     propack_exist = true;
% else
%     propack_exist = false;
%     disp("PROPACK is not correctly installed, this may slow down the initialisation step.");
%     disp("If you wish to continue anyway, press any key.");
%     disp("If you are using Linux/Mac and have PROPACK installed but still seeing this message,");
%     disp("you should replace line 40-50 to be 'propack_exist = true;'.");
%     pause;
% end
addpath E:\projects\edg\RobustNystromEDG\rpca\PROPACK\
propack_exist = true;

[m,n]     = size(D);
norm_of_D = norm(D, 'fro'); 

%% Default/Inputed parameters
max_iter  = 100;
tol       = 1e-5;
beta      = 1/(2*nthroot(m*n,4));
beta_init = 4*beta;
gamma     = 0.7;    
mu        = 5;     
trimming  = false;
show_output = 2;


if isfield(para,'beta_init') 
    beta_init = para.beta_init; 
end
if isfield(para,'beta') 
    beta = para.beta; 
end
if isfield(para,'gamma') 
    gamma = para.gamma; 
end
if isfield(para,'mu') 
    mu = para.mu; 
end
if isfield(para,'trimming') 
    trimming = para.trimming; 
end
if isfield(para,'max_iter')   
    max_iter = para.max_iter; 
end
if isfield(para,'tol')        
    tol= para.tol; 
end 
if isfield(para,'show_output') 
    show_output = para.show_output; 
end

if show_output == 2
    fprintf('beta_init = %f, beta = %f, gamma = %f, mu = [%f,%f], max_iter = %d, tol = %e\n', beta_init, beta, gamma, mu(1), mu(end), max_iter, tol);
end

err    = -1*ones(max_iter,1);
timer  = -1*ones(max_iter,1);

tic;
%%Initilization 
if propack_exist
    zeta = beta_init * lansvd(D,1); % lansvd(D,1) is the same as svds(D,1) and it provides the largest singular value of D
% else
%     zeta = beta_init * svds(D,1);
end
% When S is sparse enough, we may store S as a sparse matrix. This can save
% some memory and computing when problem size is large.
S = wthresh( D ,'h',zeta);    

if propack_exist
    [U,Sigma,V] = lansvd(D - S, r, 'L');
% else
%     [U,Sigma,V] = svds(D - S, r);
end

L = U * Sigma * V';

zeta = beta * Sigma(1,1); 
S = wthresh( D - L ,'h',zeta);

init_timer = toc;
init_err = norm(D-L-S,'fro')/norm_of_D;
if show_output == 2
    fprintf('Initialization: error: %e, timer: %f \n', init_err, init_timer);
end



%% Main Alogorithm
for t = 1 : max_iter
    tic;
    %% Trim
    if trimming
        [U, V] = trim( U, Sigma(1:r,1:r), V, mu(1), mu(end) );
    end

    %% update L
    Z = D - S;
    % These 2 QR can be computed parallelly
    [Q1,R1] = qr(Z' * U - V * ((Z * V)' * U), 0);
    [Q2,R2] = qr(Z  * V - U * ( U' * Z  * V), 0);

    M = [ U'*Z*V, R1'            ;
          R2    , zeros(size(R2)); ];
    [U_of_M, Sigma, V_of_M] = svd(M,'econ');
    % These 2 matrices multiplications can be computed parallelly
    U = [U, Q2] * U_of_M(:,1:r);
    V = [V, Q1] * V_of_M(:,1:r);
    L = U * Sigma(1:r,1:r) * V';

    %% update S
    zeta = beta * (Sigma(r+1,r+1) + (gamma^t)*Sigma(1,1));
    S = wthresh( D - L ,'h',zeta);

    %% Stop Condition
    err(t) = norm (D - L - S,'fro')/norm_of_D;
    timer(t) = toc;
    if err(t) < tol  
        if show_output >= 1
            fprintf('RPCA: Total %d iteration, final error: %e, total time without init: %f , with init: %f\n', t, err(t), sum(timer(timer>0)),sum(timer(timer>0))+init_timer);
        end
        return;
    else
        if show_output == 2
            fprintf('Iteration %d, error: %e, time: %f \n', t, err(t), timer(t));
        end
    end
end


fprintf('Maximum iterations reached, final error: %e.\n======================================\n', err(t));
end
