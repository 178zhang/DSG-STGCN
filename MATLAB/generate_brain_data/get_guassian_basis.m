function basis = get_guassian_basis(G_short_dist_mat, delta, max_num)
    Np = size(G_short_dist_mat,1);
    basis = zeros(Np, Np);
    if nargin<=1
        max_num = 15;
        delta = 10;
    end

    for ii = 1: Np
       [~, index] = sort(G_short_dist_mat(ii, :), 'ascend'); 
       basis_vect = zeros(Np, 1);
       basis_vect(index(1)) = 1;
        for jj = 2:max_num
            basis_vect(index(jj)) =   exp(-(G_short_dist_mat(ii, index(jj))^2)/(delta^2));
        end

     basis(:,ii) = basis_vect;

    end
%basis = basis';

end 