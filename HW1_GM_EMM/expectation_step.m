function [loglike, RIC] = expectation_step(ZZ, PIC, GM)
    % set the sizes from input data
    [M,N] = size(ZZ);

    % get the weights for each gaussian
    pic1 = PIC(1);
    pic2 = PIC(2);
    pic3 = PIC(3);

    % get the mixture for each mean and covariance
    gm1 = GM{1};
    gm2 = GM{2};
    gm3 = GM{3};

    % Responsibility from data point to all clusters E-step
    RIC = zeros(M, 3);
    loglike = 0;
    for j = 1:1:M

        % unnormalized weights from the r_ic equation (numerator and
        % denominator for the r_ic equation)
        zz = ZZ(j,:);
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
end