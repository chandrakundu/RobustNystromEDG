
% Test script to compare accuracy and speed of projHr1 and projHr2
% Check if the output rank is as expected
assert(rank(projHr1(F, rankR)) == rankR, 'projHr1 output rank is incorrect.');
assert(rank(projHr2(F, rankR)) == rankR, 'projHr2 output rank is incorrect.');
% Parameters
matrixSize = [200, 200]; % Size of the input matrix
rankR = 10; % Target rank
numTrials = 10; % Number of trials for timing

% Generate a random matrix
F = randn(matrixSize);

% Initialize variables to store results
timeProjHr1 = zeros(1, numTrials);
timeProjHr2 = zeros(1, numTrials);
errorProjHr1 = zeros(1, numTrials);
errorProjHr2 = zeros(1, numTrials);

% Perform multiple trials
for trial = 1:numTrials
    % Measure time and accuracy for projHr1
    tic;
    F_r1 = projHr1(F, rankR);
    timeProjHr1(trial) = toc;
    errorProjHr1(trial) = norm(F - F_r1, 'fro');

    % Measure time and accuracy for projHr2
    tic;
    F_r2 = projHr2(F, rankR);
    timeProjHr2(trial) = toc;
    errorProjHr2(trial) = norm(F - F_r2, 'fro');
end

% Display results
fprintf('Average time for projHr1: %.6f seconds\n', mean(timeProjHr1));
fprintf('Average time for projHr2: %.6f seconds\n', mean(timeProjHr2));
fprintf('Average error for projHr1: %.6f\n', mean(errorProjHr1));
fprintf('Average error for projHr2: %.6f\n', mean(errorProjHr2));

% Compare results
if mean(timeProjHr1) < mean(timeProjHr2)
    fprintf('projHr1 is faster on average.\n');
else
    fprintf('projHr2 is faster on average.\n');
end

if mean(errorProjHr1) < mean(errorProjHr2)
    fprintf('projHr1 is more accurate on average.\n');
else
    fprintf('projHr2 is more accurate on average.\n');
end

function [F_r] = projHr1(F, r)
    [U, S, V] = svds(F, r);
    F_r = U * S * V';
end


function Xr = projHr2(X,r)
    % PROJHR Projects the matrix X onto the set of matrices with rank r
    % using Singular Value Decomposition (SVD)
    [U, S, V] = svd(X, 'econ');
    S = diag(S);
    S(r+1:end) = 0; % Set all singular values after r to zero
    Xr = U * diag(S) * V'; % Reconstruct the matrix with rank r
end
