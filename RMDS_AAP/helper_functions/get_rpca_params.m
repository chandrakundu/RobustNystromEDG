function para = get_rpca_params(F, r)
% GET_RPCA_PARAMS sets the default parameters for the RPCA algorithm.

    % default parameters
    [m, n] = size(F); 
    para.mu        = 1.1*get_mu_kappa(F,r);  
    para.beta_init = r*sqrt(para.mu(1)*para.mu(end))/(sqrt(m*n));
    para.beta      = r*sqrt(para.mu(1)*para.mu(end))/(4*sqrt(m*n));
    para.trimming  = false;
    para.tol       = 1e-14;
    para.gamma     = 0.9;
    para.max_iter  = 500;
    para.show_output = 0;
end
