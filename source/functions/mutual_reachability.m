function D = mutual_reachability( D,coreDist )
    % D = mutual_reachability( D,coreDist )
    %
    % Finds the mutual reachability between points as:
    %
    %		max{ core(i),core(j),D(i,j) }
    %
    % Inputs:
    %	D - n x n distance matrix
    %	coreDist - 1 x n or n x 1 vector of core distances
    %
    % Outputs:
    %	D - n x n mutual reachability graph, edited in place
    %
    % written by Jordan Sorokin, 10/7/2017

    % find where core(i) or core(j) > D(i,j)
    maxCore = bsxfun( @max,coreDist,coreDist' );
    idx = maxCore > D; 
    D(idx) = maxCore(idx); % ~idx locations left as D

end