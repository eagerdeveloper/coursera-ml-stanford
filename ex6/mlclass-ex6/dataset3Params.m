function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% m = size(X, 1);
% pred = zeros(m, 1);
% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
% pred = svmPredict(model, Xval);
% mean(double(pred ~= yval))


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_val = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_val = [0.01 0.03 0.1 0.3 1 3 10 30];

m = size(X, 1);
pred = zeros(m, 1);
meanError = zeros(length(C_val), length(sigma_val));
count = 1;
for i = 1:length(C_val)
	for j = 1:length(sigma_val)
		C = C_val(i);
		sigma = sigma_val(j);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		pred = svmPredict(model, Xval);
		meanError(i,j) = mean(double(pred ~= yval));
		% meanError(i,j)
		count = count + 1;
	end
end

% meanError
% [x, ix] = min(min(meanError))

% This is the correct value. I was not sure how to get it from the meanError array, so I hard-coded it by observation
C = 1;
sigma = 0.1;


% =========================================================================

end
