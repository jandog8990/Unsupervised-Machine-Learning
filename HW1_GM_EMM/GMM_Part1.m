
% clc
% clear
% close all


%% 2 GMM Function.

% M = number of column vectors.
% D = dimension representing Mu_i, 1 <= i <= M.
% A set of matrices containing the corresponding covariance functions.
% And a vector pi of mixture values.

% The input should also contain an arbitrary set of vectors in the space.
% The corresponding output is their probability density
% value according to the Gaussian Mixture distribution.


%% 3 Gaussian Distribution

% In order to produce artificial data, construct a function
% whose inputs are a D dimensional vector (mean),
% a D dimensional matrix (covariance),
% and a scalar N.

% The output will produce N data according to
% the corresponding Gaussian distribution.

% It is recommended to use a circular Gaussian distribution and
% a linear transformation according to the Gaussian parameters.

% Using this distribution, generate a 2-dimensional
% distribution with 400 data, with the parameters:

mu1 = [1 2];
sigma1 = [3 1; 1 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
mu3 = [3 -3];
sigma3 = [1 0.3; 0.3 1];

%% Create the random means
a = 0; b = 5;
p = a + (b-a)*rand();
disp(p);
mu1 = mu1 + p;
mu2 = mu2 - p;
mu3 = mu3 + p;

%% Update the random sigmas
a = 0; b = 3;
p = a + (b-a)*rand();
disp(p);
sigma1 = sigma1 + p;
sigma2 = sigma2 - p;
sigma3 = sigma3 + p;


disp("Sigma 1:");
disp(sigma1);
disp("\n");

disp("Sigma 2:");
disp(sigma2);
disp("\n");

disp("Sigma 3:");
disp(sigma3);
disp("\n");

%% NOTE: Can make these weights 1/3 each (too complex here)
% PIC = [.3350 .3932 .2718];
PIC = [0.33 0.33 0.33];


% Create a gmdistribution object by using the gmditribution function.
% By default, the function creates an equal proportion mixture.
gm1 = gmdistribution(mu1,sigma1);
gm2 = gmdistribution(mu2,sigma2);
gm3 = gmdistribution(mu3,sigma3);

%% Generate PDFs from the GMM objects

% Grid of coordinates for representation in the 2-D space.
x=linspace(-6,6,30);
x=repmat(x,length(x),1);
y=x';
% Vectorization of the coordinates.
z=[x(:),y(:)];

%% Generate new pdfs based on z test data
pdf1 = pdf(gm1,z);
pdf2 = pdf(gm2,z);
pdf3 = pdf(gm3,z);
% PDF = [pdf1 pdf2 pdf3];

%% Make sure and move code from GMM_Manel1 and GMM_Part1 together
figure(6)
scatter(X,Y,10,'ko') % Plots the data
hold on
contour(x,y,buffer(pdf1,sqrt(length(pdf1)),0)) % Contours the pdf (Called 't1' here.)
contour(x,y,buffer(pdf2,sqrt(length(pdf2)),0)) % Contours the pdf (Called 't2' here.)
contour(x,y,buffer(pdf3,sqrt(length(pdf3)),0)) % Contours the pdf (Called 't3' here.)

%% Need to generate new PDFs for the ZZ data

GM = {gm1, gm2, gm3};   % cell array contains objects

% get the weights for each gaussian
pic1 = PIC(1);
pic2 = PIC(2);
pic3 = PIC(3);

% get the mixture for each mean and covariance
gm1 = GM{1};
gm2 = GM{2};
gm3 = GM{3};

% Responsibility from data point to all clusters E-step
RIC = zeros(length(X), 3);
loglike = 0;
for j = 1:1:length(X)

    % unnormalized weights from the r_ic equation (numerator and
    % denominator for the r_ic equation)
%         xx = X(j); yy = Y(j);   % get the 2D point in space
    zz = ZZtop(j,:);
    wp1 = pic1*pdf(gm1, zz);
    wp2 = pic2*pdf(gm2, zz);
    wp3 = pic3*pdf(gm3, zz);

    % total denominator sum for the r_ic equation
    den = wp1 + wp2 + wp3;

    % normalize the wp scalars
    r1 = wp1/den; r2 = wp2/den; r3 = wp3/den;
    loglike = loglike + log(r1 + r2 + r3);
    RIC(j,:) = [r1 r2 r3];
    %         RIC(j, i) = pic
end

% Maximization step
M = length(X);
m1 = sum(RIC(:,1)); m2 = sum(RIC(:,2)); m3 = sum(RIC(:,3));
pi1 = m1/M; pi2 = m2/M; pi3 = m3/M;

% Create the responsiblity vectors for each cluster
RIC1 = RIC(:,1); RIC2 = RIC(:,2);RIC3 = RIC(:,3);

% Loop through data samples and comput the new means
mu_v1 = compute_mean(m1, RIC1, ZZtop);
mu_v2 = compute_mean(m2, RIC2, ZZtop);
mu_v3 = compute_mean(m3, RIC3, ZZtop);

% Loop through data points and compute new sigmas
sigm1 = compute_sigma(RIC1, mu_v1, m1, ZZtop);
sigm2 = compute_sigma(RIC2, mu_v2, m2, ZZtop);
sigm3 = compute_sigma(RIC3, mu_v3, m3, ZZtop);

% Compute the new gaussian after updating params
new_gm1 = gmdistribution(mu_v1,sigm1);
new_gm2 = gmdistribution(mu_v2,sigm2);
new_gm3 = gmdistribution(mu_v3,sigm3);

%% Generate new pdfs based on z test data
new_pdf1 = pdf(new_gm1,z);
new_pdf2 = pdf(new_gm2,z);
new_pdf3 = pdf(new_gm3,z);

contour(x,y,buffer(new_pdf1,sqrt(length(new_pdf1)),0)) % Contours the pdf (Called 't1' here.)
contour(x,y,buffer(new_pdf2,sqrt(length(new_pdf2)),0)) % Contours the pdf (Called 't2' here.)
contour(x,y,buffer(new_pdf3,sqrt(length(new_pdf3)),0)) % Contours the pdf (Called 't3' here.)
