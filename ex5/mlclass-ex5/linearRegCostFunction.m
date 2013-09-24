function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% sizeX = size(X)           % 12 x 2
% sizeY = size(y)           % 12 x 1
% sizeTheta = size(theta)   %  2 x 1
% sizeLambda = size(lambda) %  1 x 1
% sizeGrad = size(grad)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
n = size(theta);
J=(1/(2*m)) * sum(((X*theta) - y).^2) + (lambda/(2 * m)) * sum(theta'(:,2:n).^2) ;

% X*theta % 12 x 1
% y       % 12 x 1
grad = (1/m)*(X'*(X*theta - y)) + (lambda/m)*theta;
grad(1,1) = grad(1,1) - (lambda/m)*theta(1,1);

% grad(1,1) = grad(1,1) - (lambda/m)*theta'(1,1);
% n = size(theta)
 % a = size(grad)




% =========================================================================

grad = grad(:);

end
