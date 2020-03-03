function sigma = compute_sigma(ric, mu_c, mc, ZZ)
    [M,N] = size(ZZ);
    sigma = zeros(2,2);
    for i = 1:M
        zz = ZZ(i, :);
        prod = ric(i)*(zz-mu_c)'*(zz-mu_c);
        sigma = sigma + prod;
    end
    sigma = sigma./mc;
end