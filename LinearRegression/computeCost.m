function J = computeCost(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

predictions = X*theta; % predictions of hypothesis on all m examples
sqrErrors = (predictions - y).^2; % squared errors

J = 1/(2 * m) * sum(sqrErrors);

end
