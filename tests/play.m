
function tests = test_play
    tests = functiontests(localfunctions);
end

function testHardThresholding(testCase)
    % Test 1: Basic functionality
    F = [0.1, 0.5, 0.9; -0.2, -0.6, 0.3];
    zeta = 0.4;
    expected = [0, 0.5, 0.9; 0, -0.6, 0];
    actual = hard_thresholding(F, zeta);
    verifyEqual(testCase, actual, expected);

    % Test 2: All elements below threshold
    F = [0.1, 0.2; -0.1, 0.3];
    zeta = 0.5;
    expected = zeros(size(F));
    actual = hard_thresholding(F, zeta);
    verifyEqual(testCase, actual, expected);

    % Test 3: All elements above threshold
    F = [1, 2; -3, 4];
    zeta = 0.5;
    expected = F;
    actual = hard_thresholding(F, zeta);
    verifyEqual(testCase, actual, expected);

    % Test 4: Edge case with zeta = 0
    F = [0.1, -0.2; 0.3, -0.4];
    zeta = 0;
    expected = F;
    actual = hard_thresholding(F, zeta);
    verifyEqual(testCase, actual, expected);

    % Test 5: Edge case with zeta > max(abs(F))
    F = [0.1, -0.2; 0.3, -0.4];
    zeta = 1;
    expected = zeros(size(F));
    actual = hard_thresholding(F, zeta);
    verifyEqual(testCase, actual, expected);
end

function S = hard_thresholding(F, zeta)
    % HARD_THRESHOLDING Performs hard thresholding on the matrix F
    S = F .* (abs(F) > zeta);
end
