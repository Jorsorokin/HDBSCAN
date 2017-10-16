function [delta,S_hat] = OFCH( S,parent )
    % Optimal Flat Clustering from Hierarchy
    %
    % [delta,S_hat] = OFCH( S,parentClust )
    %
    % Computes the optimal cluster scheme from a cluster hierarchy
    % cluster stability vector. 
    %
    % Inputs
    %   S - 1 x K vector of cluster stabilities, where K = # clusters
    %   
    %   parent - 1 x K vector of parent clusters of current cluster.
    %            i.e. parent(i) = parent of ith cluster
    %
    % Outputs
    %   delta - 1 x m < K vector of optimal clusters
    %
    %   S_hat - 1 x m vector of modified cluster stabilities
    %           
    %               S_hat(i) = max{ S(i), sum( S_hat(c) ) }
    %           
    %           for c in the set of all children of cluster i
    % 
    % Written by Jordan Sorokin, 10/5/2017

    % get the parent-child node pair and preallocate indicator vectors
    uniqueParent = fliplr( unique( parent(parent>0) ) );
    S_hat = S;
    delta = true( 1,numel( S ) );

    for clust = uniqueParent

        % sum stability of children of current parent
        children = (parent == clust);
        S_child = sum( S_hat(children) );

        % compare with stability of parent and store the larger
        [S_hat(clust),ind] = max( [S(clust),S_child] ); % [EQ 5] 

        % update our indicator vector
        if ind == 1
            % this parent node is stable, so set children to 0
            allChildren = get_all_subnodes( clust,parent );
            delta(allChildren) = false;
        else
            % children more stable, so set current clust to 0
            delta(clust) = false;
        end
    end
    
    delta = find( delta );
end