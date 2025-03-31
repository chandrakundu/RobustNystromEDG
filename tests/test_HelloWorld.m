classdef test_HelloWorld < matlab.unittest.TestCase
    methods(Test)
        function testHelloWorldOutput(testCase)
            expectedOutput = 'Hello, World!';
            actualOutput = HelloWorld();
            testCase.verifyEqual(actualOutput, expectedOutput);
        end
    end
end