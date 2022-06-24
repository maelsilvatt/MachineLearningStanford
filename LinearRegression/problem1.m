% Machine Learning Stanford - Linear Regression
% ==============================================================================
% Problem 1 description:
% 
% Our objective is to implement linear regression on a data example that refers
% to the amount of population (x) in 10,000s and previous data of all 
% profits (y) in $10,000
%
% ==============================================================================
% Project structure:
%
% problem1.m                  % Application of linear regression to a simple
%                             % problem
%
% problem2.m                  % Application of linear regression to a more
%                             % complex problem
%
% data1.txt                   % Data used to solve 'problem1.m'
%
% plotData.m                  % Plots data for visualization
%
% gradientDescent.m           % Calculates the optmum choice of theta using
%                             % Batch Gradient Descent
%
% computeCost.m               % Calculates the cost function using
%                             % Square Errors function
%
% featureNormalize.m          % Applies normalization to data for a better
%                             % performance
%
% normalEqn.m                 % Uses normal equation to minimize cost J
%
% ==============================================================================
%% Initialization
clear ; close all; clc

% ==============================================================================
% Part 1: Plotting data
%
fprintf('Plotting data: \n')
data = load('data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% ==============================================================================
% Part 2: Calculating Batch Gradiend Descent
%
X = [ones(m, 1), data(:,1)]; % Add a collumn of ones for vectorization
theta = zeros(2, 1); % initialize fitting parameters

% Setting gradient descent parameters:
iterations = 1500;
alpha = 0.01;

fprintf('\nTesting the cost function ...\n')

% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);


% further testing of the cost function
J = computeCost(X, y, [-1 ; 2]);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training set', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;