function err = rel_error(A_true, A_hat)
	err = norm(A_true - A_hat, "fro")/norm(A_true, "fro");
end