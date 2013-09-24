function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = size(theta);
J=(1/m) * sum(-y'.*log(sigmoid(theta'*X'))-(1-y').*log(1-sigmoid(theta'*X'))) + (lambda/(2 * m)) * sum(theta'(:,2:n).^2) ;

grad = (1/m)*((sigmoid(theta'*X')'.- y)'*X) + (lambda/m)*theta';
% n = size(theta)
 % a = size(grad)

grad(1,1) = grad(1,1) - (lambda/m)*theta'(1,1);

% grad0 = (1/m)*((sigmoid(theta'*X')'.- y)'*X(1:));
% grad1 = (1/m)*((sigmoid(theta'*X')'.- y)'*X(2:)) + (lambda/m)*theta'(:,2:n);
% grad = grad0 + grad1;




% =============================================================

end