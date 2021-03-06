function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% sizeTheta1_grad = size(Theta1_grad)
% sizeTheta2_grad = size(Theta2_grad)
% sizeX = size(X)
% sizey = size(y)
% input_layer_size
% hidden_layer_size
% num_labels
% lambda


% Add ones to the X data matrix
A1 = [ones(m, 1) X];
% sizeA1WithOne = size(A1)

Z2 = Theta1 * A1';
A2 = sigmoid(Z2);
% sizeA2 = size(A2)

A2Trans = [ones(m, 1) A2'];
% sizeA2Trans = size(A2Trans)

% Theta1 = [ones(hidden_layer_size, 1) Theta1];

Z3 = Theta2 * A2Trans';
A3 = sigmoid(Z3);
% sizeA3 = size(A3)

yK = eye(num_labels)(y,:);
% sizeyK = size(yK)

cost  = yK'.*log(A3)+(1-yK').*log(1-A3);
% sizecost = size(cost)
J = -(1/m) *sum(sum(cost,2));

regularization = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
% sizereg = size(regularization)
J = J + (lambda/(2*m)) * regularization;


delta3 = A3' - yK;
% sizedelta3 = size(delta3)

% sizeTheta2 = size(Theta2)
% sizeTheta1 = size(Theta1)

delta2 =  delta3*Theta2(:,2:end) .* sigmoidGradient(Z2');
% sizedelta2 = size(delta2)
% sizeA1 = size(A1)
% sizeA2Trans = size(A2Trans)

Delta1 = delta2'*A1;
Delta2 = delta3'*A2Trans;
% sizeDelta1 = size(Delta1)
% sizeDelta2 = size(Delta2)

Theta1_grad = (1/m) * (Delta1 + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)]);
% sizeTheta1_grad = size(Theta1_grad)
Theta2_grad = (1/m) * (Delta2 + lambda*[zeros(num_labels , 1) Theta2(:,2:end)]);
% sizeTheta2_grad = size(Theta2_grad)

% for t=1:m
% 	A1 = X(t,:);
% end

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
