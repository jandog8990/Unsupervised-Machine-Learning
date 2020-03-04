%% Execute the EM algorithm in two steps
%% 1. Generate the predictions for 1 iteration
%% 2. Gnerate consecutive predicts for N iterations

%% Generate random means for the distributions
A = min(X); B = max(X); m = 1; n = 2;
mu1 = unifrnd(A, B, m, n);
mu2 = unifrnd(A, B, m, n);
mu3 = unifrnd(A, B, m, n);

%% Generate random cov matrices for dists
A = 0.1; B = 4; m = 2; n = 2;
p1 = A+(B-A)*rand();
p2 = A+(B-A)*rand();
p3 = A+(B-A)*rand();
sigma1 = cov(ZZtop/1.8+2*p1);
sigma2 = cov(ZZtop/2.0-4*p2);
sigma3 = cov(ZZtop/2.1+6*p3);

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
gm1 = gmdistribution(mu1, sigma1);
gm2 = gmdistribution(mu2, sigma2);
gm3 = gmdistribution(mu3, sigma3);

GM = {gm1, gm2, gm3};   % cell array contains objects

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
figure
scatter(X,Y,10,'ko') % Plots the data
hold on
contour(x,y,buffer(pdf1,sqrt(length(pdf1)),0)) % Contours the pdf (Called 't1' here.)
contour(x,y,buffer(pdf2,sqrt(length(pdf2)),0)) % Contours the pdf (Called 't2' here.)
contour(x,y,buffer(pdf3,sqrt(length(pdf3)),0)) % Contours the pdf (Called 't3' here.)
title("Initialization: Random Gaussians");
hold off 

% % Represents the PDF in a 3-D plot.
% figure(1)
% hold on
% surf(x,y,buffer(pdf1,sqrt(length(pdf1)),0),'FaceColor','interp',...
%     'EdgeColor','none','FaceLighting','phong')
% surf(x,y,buffer(pdf2,sqrt(length(pdf2)),0),'FaceColor','interp',...
%     'EdgeColor','none','FaceLighting','phong')
% surf(x,y,buffer(pdf3,sqrt(length(pdf3)),0),'FaceColor','interp',...
%     'EdgeColor','none','FaceLighting','phong')
% hold off
% 
% axis tight
% view(-50,30)
% camlight left
% drawnow

%% Iterate throught the gaussians and find the best model
NN = 14;
for i = 1:NN
    [loglike, RIC] = expectation_step(ZZtop, PIC, GM);
    [PIC, GM] = maximization_step(RIC, ZZtop);
    
    %% Generate new pdfs based on z test data
    new_pdf1 = pdf(GM{1}, z);
    new_pdf2 = pdf(GM{2}, z);
    new_pdf3 = pdf(GM{3}, z);

    figure(10)
    scatter(X,Y,10,'ko');
    hold on
    contour(x,y,buffer(new_pdf1,sqrt(length(new_pdf1)),0)) % Contours the pdf (Called 't1' here.)
    contour(x,y,buffer(new_pdf2,sqrt(length(new_pdf2)),0)) % Contours the pdf (Called 't2' here.)
    contour(x,y,buffer(new_pdf3,sqrt(length(new_pdf3)),0)) % Contours the pdf (Called 't3' here.)
    title("EM Algorithm Iterations");
    hold off
end