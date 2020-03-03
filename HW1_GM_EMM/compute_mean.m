function mu_v = compute_mean(mc, ric, ZZ)
    [M, N] = size(ZZ);
    mu_v = zeros(1,2);
    for i = 1:M
        zz = ZZ(i, :);
        prod = ric(i)*zz;
        mu_v = mu_v + prod;
    end
    mu_v = mu_v./mc;
end