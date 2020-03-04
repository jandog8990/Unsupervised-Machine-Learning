%% Generate the random Gaussian samples
clear all
close all

mu1 = [3 2];
sigma1 = [3 1; 1 2];
mu2 = [-4 -2];
sigma2 = [2 0; 0 1];
mu3 = [3 -3];
sigma3 = [1 0.2; 0.2 1];

% mu1 = [1 2];
% sigma1 = [3 1; 1 2];
% mu2 = [-1 -2];
% sigma2 = [2 0; 0 1];
% mu3 = [3 -3];
% sigma3 = [1 0.3; 0.3 1];

% Mu1 and Sigma1:

% This may be the first random Gaussian
z = randn(2,1000); % Gaussian data with zero mean and covariance |
% trueMean_1 = [1, 2]';
% trueCovar_1 = [3 1; 1 2];

% Calculate 
X1 = gaussian_distribution(z, sigma1, mu1');
Mean_1 = mean(X1,2);
Covariance_1 = cov(X1');

X2 = gaussian_distribution(z, sigma2, mu2');
Mean_2 = mean(X2,2);
Covariance_2 = cov(X2');

X3 = gaussian_distribution(z, sigma3, mu3');
Mean_3 = mean(X3,2);
Covariance_3 = cov(X3');

disp("True 1:");
disp(Mean_1)
disp(Covariance_1)

disp("True 2:");
disp(Mean_2)
disp(Covariance_2)

disp("True 3:");
disp(Mean_3)
disp(Covariance_3)

%% 5.1 EM Algorithm

% Write a script that iteratively updates the parameters
% of the GMM using the EM algorithm. For representation
% of the results, use the function contour of MATLAB.

% For representation purposes, you can use the following ocde:

X = [X1(1,:) X2(1,:) X3(1,:)];
Y = [X1(2,:) X2(2,:) X3(2,:)];
ZZtop = [X(:), Y(:)];   % your main data matrix of X,Y coords
disp("ZZ Size:");
disp(size(ZZtop));

figure
scatter(X,Y,10,'ko') % Plots the data
