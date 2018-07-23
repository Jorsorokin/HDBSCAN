function [dCore,D,kdtree] = compute_core_distances( X,k )
    % [dCore,D,kdtree] = compute_core_distances( X,k )
    %
    % computes the core distances of each point in X as:
    %   
    %       dCore(i) = D(i,j); j=k, the kth neareast neighbor of X(i)
    %
    % Inputs:
    %   X - n x m matrix, n = observations, m = variables (dimensions)
    %
    %   k - scalar specifying the number of nearest neighbors 
    %
    % Outputs:
    %   dCore - n x 1 vector of core distances 
    %
    %   D - n x n matrix of pairwise distances if m > 10, or nan otherwise.
    %       In the case of m <= 10, the k-nearest neighbor search is
    %       performed more efficiently using KD trees, thus the n x n
    %       distance matrix is not necessary 
    %
    %   kdtree - if m <= 10 and n > 100, kdtree is a KDTree object of X, 
    %            allowing for fast nearest-neighbor queries in the future.
    %            Else, kdtree = nan;
    %
    % Written by Jordan Sorokin, 10/6/2017

    [n,m] = size( X );
    
    if k == 1
        dCore = zeros( 1,n );
        D = nan;
        return
    end
    
    %if (m >= 10) || (n < 100)
        kdtree = nan;
        if n > 20e3
            [~,dCore,D] = compute_nearest_neighbors( X,X,k-1 ); % slow but memory conservative
        else
            D = compute_pairwise_dist( X );
            dCore = sort( D,1 );
            dCore = dCore(k,:);
        end
    %else
%         kdtree = createns( X,'nsmethod','kdtree' );
%         [neighbors,dCore] = kdtree.knnsearch( X,'k',k );
%         dCore = dCore(:,end);
%         neighbors = neighbors(:,end);
%         D = sparse( double( neighbors ),1:n,dCore,n,n );
    %end
end