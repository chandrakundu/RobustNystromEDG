function num_negative_eig = count_negative_eigenvalues(A)
    % Calculate the eigenvalues of the matrix A
    eigenvalues = eig(A);
    
    % Count the number of negative eigenvalues
    num_negative_eig = sum(real(eigenvalues) < 0);
end