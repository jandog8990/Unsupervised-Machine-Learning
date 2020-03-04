function [PIC, GM] = maximization_step(RIC, ZZ)
    % Maximization step
    [M, N] = size(ZZ);
    m1 = sum(RIC(:,1)); m2 = sum(RIC(:,2)); m3 = sum(RIC(:,3));
    pi1 = m1/M; pi2 = m2/M; pi3 = m3/M;
    
    % set the new PIC
    PIC = [pi1 pi2 pi3];

    % Create the responsiblity vectors for each cluster
    RIC1 = RIC(:,1); RIC2 = RIC(:,2); RIC3 = RIC(:,3);

    % Loop through data samples and comput the new means
    mu_v1 = compute_mean(m1, RIC1, ZZ);
    mu_v2 = compute_mean(m2, RIC2, ZZ);
    mu_v3 = compute_mean(m3, RIC3, ZZ);

    % Loop through data points and compute new sigmas
    sigm1 = compute_sigma(RIC1, mu_v1, m1, ZZ);
    sigm2 = compute_sigma(RIC2, mu_v2, m2, ZZ);
    sigm3 = compute_sigma(RIC3, mu_v3, m3, ZZ);

    % Compute the new gaussian after updating params
    new_gm1 = gmdistribution(mu_v1,sigm1);
    new_gm2 = gmdistribution(mu_v2,sigm2);
    new_gm3 = gmdistribution(mu_v3,sigm3);
    
    GM = {new_gm1 new_gm2 new_gm3};
end