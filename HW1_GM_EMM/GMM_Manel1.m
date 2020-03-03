%% EM
clc
clear all
close all

mu1 = [1 2];
sigma1 = [3 1; 1 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
mu3 = [3 -3];
sigma3 = [1 0.3; 0.3 1];

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

% % calculate Q & A (eigenvectors and eigenvalues for the cluster)
% [Q, A] = eig(trueCovar_1);
% disp("Q vec:");
% disp(Q);
% disp("A val:");
% disp(A);
% disp("\n");
% 
% % calculate the new estimated cov matrix using Q*A*Q'
% new_sigma_1 = Q*A*Q';
% disp("New Sigma 1:");
% disp(new_sigma_1);
% disp("\n")



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

figure(4)
scatter(X,Y,5,'ko') % Plots the data
hold on
contour(x,y,buffer(t1,sqrt(length(t1)),0)) % Contours the pdf (Called 't1' here.)
contour(x,y,buffer(t2,sqrt(length(t2)),0)) % Contours the pdf (Called 't2' here.)
contour(x,y,buffer(t3,sqrt(length(t3)),0)) % Contours the pdf (Called 't3' here.)
hold off
% Represents the PDF in a 3-D plot.
figure(5)
hold on
surf(x,y,buffer(t1,sqrt(length(t1)),0),'FaceColor','interp',...
    'EdgeColor','none','FaceLighting','phong')
surf(x,y,buffer(t2,sqrt(length(t2)),0),'FaceColor','interp',...
    'EdgeColor','none','FaceLighting','phong')
surf(x,y,buffer(t3,sqrt(length(t3)),0),'FaceColor','interp',...
    'EdgeColor','none','FaceLighting','phong')
hold off

axis tight
view(-50,30)
camlight left
drawnow


% For gm2:

% Grid of coordinates for representation in the 2-D space.
% x=linspace(-6,6,30);x=repmat(x,length(x),1);
% y=x';
% Vectorization of the coordinates.
% z=[x(:),y(:)];

%%% Add here the code to compute the pdf 't' at each point of 'z'.
% t2 = pdf(gm2,z);

% figure(6)
% scatter(x(1,:),x(2,:),10,'ko') % Plots the data
% hold on
% contour(x,y,buffer(t2,sqrt(length(t2)),0)) % Contours the pdf (Called 't' here.)
% hold off
% Represents the PDF in a 3-D plot.
% figure(7)
% surf(x,y,buffer(t2,sqrt(length(t2)),0),'FaceColor','interp',...
%    'EdgeColor','none','FaceLighting','phong')

% axis tight
% view(-50,30)
% camlight left
% drawnow


% For gm3:

% Grid of coordinates for representation in the 2-D space.
% x=linspace(-6,6,30);x=repmat(x,length(x),1);
% y=x';
% Vectorization of the coordinates.
% z=[x(:),y(:)];

%%% Add here the code to compute the pdf 't' at each point of 'z'.
% t3 = pdf(gm3,z);

% figure(8)
% scatter(x(1,:),x(2,:),10,'ko') % Plots the data
% hold on
% contour(x,y,buffer(t3,sqrt(length(t3)),0)) % Contours the pdf (Called 't' here.)
% hold off
% Represents the PDF in a 3-D plot.
% figure(9)
% surf(x,y,buffer(t3,sqrt(length(t3)),0),'FaceColor','interp',...
%    'EdgeColor','none','FaceLighting','phong')

% axis tight
% view(-50,30)
% camlight left
% drawnow


%% 5.2 Representation of the Data Likelihood


%% 5.3 Unsupervised Classification


%% 5.4 Comparison to the K-Means Algorithm


%% 5.4.1 Mahalanobis Distance


%% 5.4.2 K-Means Procedure


%% 5.4.3 Parameter Update


%% 5.4.4 Parameter Initialization, Growing-Prunning and Use in a 10-Dimensional Problem


%% Function Declarations.

% Used in Section 4 above


%% Compute PDF Values.

% Create a gmdistribution object and compute its pdf values.
% Define the distribution parameters (means and covariances) of a
% two-component bivariate Gaussian mixture distribution.
% mu1 = [1 2;-3 -5];
% sigma1 = [1 1]; % shared diagonal covariance matrix
% r = normrnd(mu1,sigma1)

% Create a gmdistribution object by using the gmditribution function. By
% default, the function creates an equal proportion mixture.
% gm = gmdistribution(mu1,sigma1)
% size_gm = size(gm)
% Compute the pdf values of gm.
% X = [0 0;1 2;3 3;5 3];
% Y = pdf(gm,X)
% size_Y = size(Y)


%% Plot pdf.

% Create a gmdistribution object and plot its pdf.
% Define the distribution parameters (means,covariances, and mixing
% proportions) of two bivariate Gaussian mixture components.
% p = [0.4 0.6];                  % Mixing proportions.
% mu1 = [1 2;-3 -5];              % Means.
% sigma1 = cat(3,[2 0.5],[1 1])   % Covariances 1-by-2-by-2 array.

% Create a gmdistribution object by using the gmdistribution function.
% gm = gmdistribution(mu1,sigma1)

% Plot the pdf of the Gaussian mixture distribution by using fsurf.
% fsurf(@(x,y)reshape(pdf(gm,[x(:),y(:)]),size(x)),[-5 10])

