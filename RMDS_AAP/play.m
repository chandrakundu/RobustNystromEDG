clc; clear;
alpha=0.1;
[D_obs, X_true, P_true, D_true] = generate_data_RDMS(alpha);

r = 2;
tol = 1e-6;
zeta0 = 1.2 * max(D_true(:));
gamma = 0.5;
maxIter = 50;

params.show_output = 2;
params.max_iter = maxIter;
params.r = 2;
params.zeta0 = zeta0;
params.gamma = gamma;
params.tol = tol;


data = struct( ...
        'D_obs', D_obs, ... 
        'X_true', X_true, ...
        'P_true', P_true, ...
        'D_true', D_true ...
    );

[Lk, Sk, Xk] = rd2(data, params);
