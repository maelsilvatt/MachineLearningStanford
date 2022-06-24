function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Initializing some values:
X_norm = X;
m = length(X);
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

for iter=1:m
  feature = X(:,iter);
  mu = mean(feature);
  sigma = std(feature);
  X_norm(:,iter) = (feature - mu) ./ std;
endfor







% ============================================================

end
