function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
A1 = [ones(m, 1) X];
% sizeA1WithOne = size(X)

Z2 = Theta1 * A1';
A2 = sigmoid(Z2);

% sizeA2 = size(A2)

A2Trans = [ones(m, 1) A2'];
% Theta1 = [ones(num_labels, 1) Theta1];

% sizeA2Trans = size(A2Trans)

Z3 = Theta2 * A2Trans';
A3 = sigmoid(Z3);

% sizeA3 = size(A3)

[m, im] = max(sigmoid(A3));
% n = size(im)

p = im';






% =========================================================================


end
