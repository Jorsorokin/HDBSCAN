function D = compute_pairwise_dist( X,varargin )
    % D = compute_pairwise_dist( X,(Y) )
    %
    % computes a (fast!) pairwise euclidean distance matrix of the inputs.
    % If more than one matrix is provided, will compute pairwise distances
    % between matrices. Else, computes a symmetric pairwise distance of X
    %
    % Both X and the optional matrix Y must have points along the rows, and
    % dimensions of points along columns.

    if nargin == 1
        d = sum( X.^2,2 ); 
        D = real( sqrt( bsxfun( @plus,d,d' ) - 2*(X * X') ) );
    else
        Y = varargin{1};
        if size( Y,2 ) ~= size( X,2 )
            error( 'X and Y must have equal number of dimensions' );
        end

        D = real( sqrt( bsxfun( @plus, sum( X.^2,2 ),...
            bsxfun( @minus, sum( Y.^2,2 )', 2*(X * Y') ) ) ) );
    end
end