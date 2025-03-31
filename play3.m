clear;
m = 10;
n = 10;
alpha = 0.2;
S_supp_idx = randsample(m*n, round(alpha*m*n), false);
S_range = 1*mean(mean(abs(10)));
S_temp = 2*S_range*rand(m,n)-S_range; 
S_true = zeros(m, n);
S_true(S_supp_idx) = S_temp(S_supp_idx); 


[U, S, V] = svd(randn(m,n)); 

US = U * S;