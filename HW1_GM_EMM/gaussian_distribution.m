%% Gaussian distribution function
function x = gaussian_distribution(x, S, mu)
    [Q, A] = eig(S);
    x = Q * sqrt(A) * x + mu;
end

