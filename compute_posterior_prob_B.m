function Pp_B = compute_posterior_prob_B(bma)
    % Get posterior mean and covariance
    Ep = bma.Ep;
    Cp = bma.Cp;

    % Get size of B matrix
    Bmat = Ep.B;
    [n, ~] = size(Bmat);

    % Initialize posterior probabilities
    Pp_B = NaN(n, n);

    % Flatten index map
    pvec = spm_vec(Ep);
    v = diag(spm_vec(Cp));

    for i = 1:n
        for j = 1:n
            % Get index of B(i,j) in the vectorized parameter set
            idx = spm_fieldindices(Ep, sprintf('B{1}(%d,%d)', i, j));
            

            if isempty(idx)
                continue;  % parameter not included in any model
            end

            mu = pvec(idx);
            sigma = sqrt(v(idx));

            % Compute posterior probability it's non-zero
            Pp_B(i, j) = 1 - normcdf(0, abs(mu), sigma);
        end
    end
end